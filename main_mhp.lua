--[[ 

Generic training script for MHP GAWWN keypoints,txt -> image.

--]]
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'cudnn'
util = paths.dofile('util.lua')

opt = {
  num_holdout = 1,
  dropout = 0.8,
  numCaption = 3,
  activationG = '',
  activationD = '',
  txtSize = 1024,
  fake_score_thresh = 0.1,
  doc_length = 201,
  trainfiles = '',
  cls_weight = 0.5,
  port = 8000,
  dbg = 0,
  num_elt = 16,
  keypoint_dim = 16,
  save_every = 10,
  print_every = 1,
  dataset = 'mhp',
  img_dir = '',
  filenames = '',
  data_root = '/mnt/brain3/datasets/txt2img/mhp/t7files',
  checkpoint_dir = '/home/reedscot/checkpoints',
  batchSize = 64,
  loadSize = 150,
  nclass = 20,         -- #  of dim for raw text.
  fineSize = 128,
  nt = 128,               -- #  of dim for text features.
  nz = 100,               -- #  of dim for Z
  ngf = 128,              -- #  of gen filters in first conv layer
  ndf = 64,               -- #  of discrim filters in first conv layer
  nThreads = 4,           -- #  of data loading threads to use
  niter = 1000,           -- #  of iter at starting learning rate
  lr = 0.0002,            -- initial learning rate for adam
  lr_decay = 0.5,         -- initial learning rate for adam
  decay_every = 100,
  beta1 = 0.5,            -- momentum term of adam
  ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
  display = 1,            -- display samples while training. 0 = false
  display_id = 10,        -- display window id.
  gpu = 2,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
  name = 'vg',
  noise = 'normal',       -- uniform / normal
  init_g = '',
  init_d = '',
  init_t = '',
  use_cudnn = 1,
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end
if opt.display then
  disp = require 'display' 
  disp.configure({hostname='0.0.0.0', port=opt.port})
end

assert(opt.keypoint_dim == 16 or opt.keypoint_dim == 8)

if opt.gpu > 0 then
  ok, cunn = pcall(require, 'cunn')
  ok2, cutorch = pcall(require, 'cutorch')
  cutorch.setDevice(opt.gpu)
end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------
local function weights_init(m)
  local name = torch.type(m)
  if name:find('Convolution') then
    m.weight:normal(0.0, 0.02)
    m.bias:fill(0)
  elseif name:find('BatchNormalization') then
    if m.weight then m.weight:normal(1.0, 0.02) end
    if m.bias then m.bias:fill(0) end
  end
end

local nc = 3
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

local function activationG()
  if opt.activationG == 'elu' then
    return nn.ELU()
  else
    return nn.ReLU(true)
  end
end

local function activationD()
  if opt.activationD == 'elu' then
    return nn.ELU()
  else
    return nn.LeakyReLU(0.2, true)
  end
end

if opt.init_g == '' then
  -- noise + txt encoder
  prep_noise = nn.Sequential()
    :add(nn.View(-1,opt.nz))
    :add(nn.Linear(opt.nz, ngf*4))
  prep_txt = nn.Sequential()
    :add(nn.View(-1,opt.txtSize))
    :add(nn.Linear(opt.txtSize, ngf*4))
  noise_txt = nn.Sequential()
    :add(nn.ParallelTable()
      :add(prep_noise)       -- ngf * 4
      :add(prep_txt))        -- ngf * 4
    :add(nn.JoinTable(2))
    :add(activationG())
  noise_txt_region = nn.Sequential()
    :add(nn.ParallelTable()
      :add(nn.Sequential()
        :add(noise_txt)      -- ngf * 8
        :add(nn.Linear(ngf * 8, ngf * 4))
        :add(nn.Replicate(opt.keypoint_dim,3))
        :add(nn.Replicate(opt.keypoint_dim,4)))
      :add(nn.Sequential() -- keypoints
        :add(nn.Sum(2))
        :add(nn.Clamp(0,1))
        :add(nn.Replicate(ngf*4,2))))
    :add(nn.CMulTable()) -- ngf * 4
    :add(SpatialFullConvolution(ngf * 4, ngf * 4, 3, 3, 1, 1, 1, 1))
    :add(SpatialBatchNormalization(ngf * 4)):add(activationG())
    :add(nn.ConcatTable()
      :add(nn.Sequential()
        :add(SpatialConvolution(ngf * 4, ngf, 1, 1, 1, 1, 0, 0))
        :add(SpatialBatchNormalization(ngf)):add(activationG())
        :add(SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1))
        :add(SpatialBatchNormalization(ngf)):add(activationG())
        :add(SpatialConvolution(ngf, ngf * 4, 3, 3, 1, 1, 1, 1))
        :add(SpatialBatchNormalization(ngf * 4)))
      :add(nn.Identity()))
    :add(nn.CAddTable()):add(activationG())
  prep_loc_global = nn.Sequential()
    -- (opt.num_elt) x 16 x 16
    :add(SpatialConvolution(opt.num_elt, ngf, 4, 4, 2, 2, 1, 1))
    :add(SpatialBatchNormalization(ngf)):add(activationG())
    -- (ngf) x 8 x 8
    :add(SpatialConvolution(ngf, ngf*2, 4, 4, 2, 2, 1, 1))
    :add(SpatialBatchNormalization(ngf*2)):add(activationG())
    -- (ngf) x 4 x 4
    :add(SpatialConvolution(ngf*2, ngf*4, 4, 4))
    :add(SpatialBatchNormalization(ngf*4)):add(activationG())
    :add(nn.View(-1,ngf*4))
  noise_txt_global = nn.Sequential()
    :add(nn.ParallelTable()
      :add(nn.Sequential()
        :add(noise_txt)      -- ngf * 8
        :add(nn.Linear(ngf * 8, ngf * 4))
        :add(nn.BatchNormalization(ngf * 4)):add(activationG()))
      :add(prep_loc_global))  -- ngf * 4
    :add(nn.JoinTable(2))
    :add(nn.View(-1, ngf*8, 1, 1))
    -- 1 x 1
    :add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4))
    :add(SpatialBatchNormalization(ngf * 4)):add(activationG())
    -- 4 x 4
    :add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
    :add(SpatialBatchNormalization(ngf * 2)):add(activationG())
    -- 8 x 8
    :add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
    :add(SpatialBatchNormalization(ngf)):add(activationG())
    -- 16 x 16
  -- merge with keypoints
  netG = nn.Sequential()
    :add(nn.ConcatTable()
      :add(noise_txt_global)   -- ngf
      :add(noise_txt_region)   -- ngf * 4
      :add(nn.SelectTable(2))) -- keypoints
    :add(nn.JoinTable(2))
    :add(nn.Contiguous())
    -- state size: (ngf*4 + opt.num_elt) x 16 x 16
    :add(SpatialFullConvolution(ngf * 5 + opt.num_elt, ngf * 4, 3, 3, 1, 1, 1, 1))
    :add(SpatialBatchNormalization(ngf * 4)):add(activationG())
    :add(SpatialFullConvolution(ngf * 4, ngf * 4, 3, 3, 1, 1, 1, 1))
    :add(SpatialBatchNormalization(ngf * 4))
    -- state size: (ngf*4) x 16 x 16
    local conc = nn.ConcatTable()
    local conv = nn.Sequential()
    conv:add(SpatialConvolution(ngf * 4, ngf, 1, 1, 1, 1, 0, 0))
    conv:add(SpatialBatchNormalization(ngf)):add(activationG())
    conv:add(SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1))
    conv:add(SpatialBatchNormalization(ngf)):add(activationG())
    conv:add(SpatialConvolution(ngf, ngf * 4, 3, 3, 1, 1, 1, 1))
    conv:add(SpatialBatchNormalization(ngf * 4))
    conc:add(nn.Identity())
    conc:add(conv)
    netG:add(conc)
  if opt.fineSize == 128 then
    netG:add(nn.CAddTable()):add(activationG())
      -- state size: (ngf*4) x 16 x 16
      :add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
      :add(SpatialBatchNormalization(ngf * 2)):add(activationG())
      -- state size: (ngf * 2) x 32 x 32
      :add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
      :add(SpatialBatchNormalization(ngf)):add(activationG())
      -- state size: (ngf) x 64 x 64
      :add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
      :add(nn.Tanh())
      -- state size: (nc) x 128 x 128
  elseif opt.fineSize == 64 then
    netG:add(nn.CAddTable()):add(activationG())
      -- state size: (ngf*4) x 16 x 16
      :add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
      :add(SpatialBatchNormalization(ngf * 2)):add(activationG())
      -- state size: (ngf * 2) x 32 x 32
      :add(SpatialFullConvolution(ngf * 2, nc, 4, 4, 2, 2, 1, 1))
      :add(nn.Tanh())
      -- state size: (nc) x 64 x 64
  else
    assert(false)
  end
  netG:apply(weights_init)
else
  netG = torch.load(opt.init_g)
end

if opt.init_d == '' then
  -- netD expects {img, loc, txt}
  if opt.fineSize == 128 then
    imgGlobalD = nn.Sequential()
      :add(nn.SelectTable(1))
      -- state size: (nc) x 128 x 128
      :add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
      :add(activationD())
      -- state size: (nc) x 64 x 64
      :add(SpatialConvolution(ndf, ndf, 4, 4, 2, 2, 1, 1))
      :add(activationD())
  elseif opt.fineSize == 64 then
    imgGlobalD = nn.Sequential()
      :add(nn.SelectTable(1))
      -- state size: (nc) x 64 x 64
      :add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
      :add(activationD())
  else
    assert(false)
  end
  imgGlobalD
    -- state size: (ndf) x 32 x 32
    :add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
    :add(SpatialBatchNormalization(ndf * 2)):add(activationD())
    -- state size: (ndf*2) x 16 x 16
    :add(SpatialConvolution(ndf * 2, ndf * 2, 3, 3, 1, 1, 1, 1))
    :add(SpatialBatchNormalization(ndf * 2)):add(activationD())
    -- state size: (ndf*2) x 16 x 16
  prep_txt_d = nn.Sequential()
    :add(nn.SelectTable(3))
    :add(nn.Linear(opt.txtSize, opt.nt))
    :add(activationD())
  -- region pathway
  imgTextGlobalD = nn.Sequential()
    :add(nn.ConcatTable()
      :add(imgGlobalD)   -- ndf * 2
      :add(nn.Sequential()       -- text path
        :add(prep_txt_d) -- opt.nt
        :add(nn.Replicate(opt.keypoint_dim,3))
        :add(nn.Replicate(opt.keypoint_dim,4))))
    :add(nn.JoinTable(2))
    :add(SpatialConvolution(ndf * 2 + opt.nt, ndf * 2, 3, 3, 1, 1, 1, 1))
    :add(SpatialBatchNormalization(ndf * 2)):add(activationD())
  keyMulD = nn.Sequential()
    :add(nn.ConcatTable() -- keypoint multiplication
      :add(imgTextGlobalD)
      :add(nn.Sequential()  -- keypoints
        :add(nn.SelectTable(2))
        :add(nn.Sum(2))
        :add(nn.Clamp(0,1))
        :add(nn.Replicate(ndf * 2, 2))))
    :add(nn.CMulTable())
  regionD = nn.Sequential()
    :add(nn.ConcatTable()
      :add(keyMulD)               -- (ndf*2) features with keypoint attention
      :add(nn.SelectTable(2)))    -- (opt.num_elt) keypoints
    :add(nn.JoinTable(2))         -- keypoint concatenation
    :add(nn.Contiguous())
    -- state size: (ndf*2) x 16 x 16
    :add(SpatialConvolution(ndf * 2 + opt.num_elt, ndf * 2, 1, 1))
    :add(SpatialBatchNormalization(ndf * 2)):add(activationD())
    -- state size: (ndf*2) x 16 x 16
    :add(SpatialConvolution(ndf * 2, ndf, 1, 1))
    -- state size: (ndf) x 16 x 16
    :add(nn.Mean(4))
    :add(nn.Mean(3))
    :add(activationD())
    -- global pathway
    -- state size: (ndf*2) x 16 x 16
  convGlobalD = nn.Sequential()
    :add(imgGlobalD)
    -- (ndf*2) x 16 x 16
    :add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
    :add(SpatialBatchNormalization(ndf * 4)):add(activationD())
    -- (ndf*4) x 8 x 8
    :add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
    :add(SpatialBatchNormalization(ndf * 8)):add(activationD()) -- now 4x4
    -- (ndf*8) x 4 x 4
  txtGlobalD = nn.Sequential()
    :add(prep_txt_d)
    :add(nn.Replicate(4,3))
    :add(nn.Replicate(4,4))
  globalD = nn.Sequential()
    :add(nn.ConcatTable()
      :add(convGlobalD)
      :add(txtGlobalD))
    :add(nn.JoinTable(2))
    :add(nn.Contiguous())
    -- state size: (ndf*8 + opt.nt) x 4 x 4
    :add(SpatialConvolution(ndf * 8 + opt.nt, ndf * 4, 1, 1))
    :add(SpatialBatchNormalization(ndf * 4))
    :add(activationD())
    -- state size: (ndf*4) x 4 x 4
    :add(SpatialConvolution(ndf * 4, ndf, 4, 4, 1, 1))
    :add(SpatialBatchNormalization(ndf)):add(activationD())
    -- state size: (ndf) x 1 x 1
    :add(nn.View(-1,ndf))
    :add(nn.Dropout(opt.dropout))
  netD = nn.Sequential()
    :add(nn.ConcatTable()
      :add(regionD)
      :add(globalD))
    :add(nn.JoinTable(2))
    :add(nn.Linear(ndf * 2, ndf))
    :add(nn.BatchNormalization(ndf)):add(activationD())
    :add(nn.Linear(ndf, 1))
    :add(nn.Sigmoid())
  netD:apply(weights_init)
else
  netD = torch.load(opt.init_d)
end

netT = torch.load(opt.init_t)

local criterion = nn.BCECriterion()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
local dict = {}
for i = 1,#alphabet do
  dict[alphabet:sub(i,i)] = i
end
ivocab = {}
for k,v in pairs(dict) do
  ivocab[v] = k
end
alphabet_size = #alphabet
----------------------------------------------------------------------------
local input_img = torch.zeros(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local input_fake = torch.zeros(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local input_loc = torch.zeros(opt.batchSize, opt.num_elt, opt.keypoint_dim, opt.keypoint_dim)
local input_txt = torch.Tensor(opt.batchSize, opt.txtSize)
local input_txt_shuf = torch.Tensor(opt.batchSize, opt.txtSize)
local noise = torch.zeros(opt.batchSize, nz)
local label = torch.zeros(opt.batchSize)
local errD, errG
----------------------------------------------------------------------------
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   input_img = input_img:cuda()
   input_txt = input_txt:cuda()
   input_txt_shuf = input_txt_shuf:cuda()
   input_fake = input_fake:cuda()
   input_loc = input_loc:cuda()
   noise = noise:cuda()
   label = label:cuda()
   netD:cuda()
   netG:cuda()
   netT:cuda()
   criterion:cuda()
end

if (opt.gpu >= 0) and (opt.use_cudnn == 1) then
  cudnn = require('cudnn')
  netD = cudnn.convert(netD, cudnn)
  netG = cudnn.convert(netG, cudnn)
  netT = cudnn.convert(netT, cudnn)
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end

local sample = function()
  data_tm:reset(); data_tm:resume()
  real_img, real_txt, real_loc, dbg, loc_raw = data:getBatch()
  data_tm:stop()

  input_img:copy(real_img)
  input_txt:copy(real_txt)
  input_loc:copy(real_loc)

  local shuf_ix = torch.randperm(opt.batchSize)
  for n = 1,input_txt:size(1) do
    input_txt_shuf[n]:copy(input_txt[shuf_ix[n]])
  end
end

-- create closure to evaluate f(X) and df/dX of discriminator
fake_score = 0.5
local fDx = function(x)
  gradParametersD:zero()

  -- train with real
  label:fill(real_label)
  local output = netD:forward{input_img, input_loc, input_txt}
  errD_real = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  local deltas = netD:backward({input_img, input_loc, input_txt}, df_do)

  -- train with wrong
  errD_wrong = 0
  if opt.cls_weight > 0 then
    -- train with wrong 
    label:fill(fake_label)
    local output = netD:forward{input_img, input_loc, input_txt_shuf}
    errD_wrong = opt.cls_weight*criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    df_do:mul(opt.cls_weight)
    deltas = netD:backward({input_img, input_loc, input_txt_shuf}, df_do)
  end

  -- train with fake
  if opt.noise == 'uniform' then -- regenerate random noise
    noise:uniform(-1, 1)
  elseif opt.noise == 'normal' then
    noise:normal(0, 1)
  end

  label:fill(fake_label)
  local fake = netG:forward({{noise, input_txt}, input_loc})
  input_img:copy(fake)
  local output = netD:forward{input_img, input_loc, input_txt}
  -- update fake score tracker
  local cur_score = output:mean()
  fake_score = 0.99 * fake_score + 0.01 * cur_score
  local errD_fake = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  local fake_weight = 1 - opt.cls_weight
  errD_fake = errD_fake*fake_weight
  df_do:mul(fake_weight)
  netD:backward({input_img, input_loc, input_txt}, df_do)

  errD = errD_real + errD_fake + errD_wrong
  return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
  gradParametersG:zero()

  label:fill(real_label) -- fake labels are real for generator cost
  local output = netD.output

  local cur_score = output:mean()
  fake_score = 0.99 * fake_score + 0.01 * cur_score

  errG = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  local df_dr = netD:updateGradInput({input_img, input_loc, input_txt}, df_do)
  local deltas = netG:backward({{noise, input_txt}, input_loc}, df_dr[1])

  return errG, gradParametersG
end

-- train
for epoch = 1, opt.niter do
  epoch_tm:reset()

  if epoch % opt.decay_every == 0 then
    optimStateG.learningRate = optimStateG.learningRate * opt.lr_decay
    optimStateD.learningRate = optimStateD.learningRate * opt.lr_decay
  end

  for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
    tm:reset()

    sample()
    if fake_score > opt.fake_score_thresh then
      optim.adam(fDx, parametersD, optimStateD)
    else
      -- just do fDx, no update.
      fDx(parametersD)
    end
    optim.adam(fGx, parametersG, optimStateG)

    -- logging
    if ((i-1) / opt.batchSize) % opt.print_every == 0 then
      print(('[%d][%d/%d] T:%.3f  DT:%.3f lr: %.4g '
                .. ' G:%.3f  D:%.3f fs:%.2f'):format(
              epoch, ((i-1) / opt.batchSize),
              math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
              tm:time().real, data_tm:time().real,
              optimStateG.learningRate,
              errG and errG or -1, errD and errD or -1,
              fake_score))
      local fake = netG.output
      disp.image(fake:narrow(1,1,math.min(4,opt.batchSize)), {win=opt.display_id, title=opt.name})
      local vis_real = real_img:narrow(1,1,math.min(4,opt.batchSize))
      for b = 1,vis_real:size(1) do
        vis_real[b] = util.draw_keypoints(vis_real[b], loc_raw[b])
      end
      disp.image(vis_real, {win=opt.display_id * 3, title=opt.name})
      local tmp = input_loc:clone():max(2)
      tmp = torch.repeatTensor(tmp, 1, 3, 1, 1)
      disp.image(tmp:narrow(1,1,math.min(4,opt.batchSize)), {win=opt.display_id * 7, title=opt.name})
    end
  end
  if epoch % opt.save_every == 0 then
    paths.mkdir(opt.checkpoint_dir)
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG:clone():clearState())
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD:clone():clearState())
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_opt.t7', opt)
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
           epoch, opt.niter, epoch_tm:time().real))
  end
end

