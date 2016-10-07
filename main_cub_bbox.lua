--[[ 

Generic training script for CUB GAWWN.

--]]
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'stn'
require 'cudnn'
require 'CropInvert'
util = paths.dofile('util.lua')

opt = {
  num_holdout = 0,
  numCaption = 1,
  port = 8000,
  iou_thresh = 0.8,
  dbg = 0,
  num_elt = 2,
  save_every = 10,
  print_every = 1,
  dataset = 'cub_stn',
  img_dir = '',
  cls_weight = 0.5,
  filenames = '',
  data_root = '/mnt/brain3/datasets/txt2img/cub_ex_loc',
  checkpoint_dir = '/home/reedscot/checkpoints',
  batchSize = 64,
  doc_length = 201,
  loadSize = 150,
  txtSize = 1024,         -- #  of dim for raw text.
  fineSize = 128,
  nt = 128,               -- #  of dim for text features.
  nz = 100,               -- #  of dim for Z
  ngf = 128,              -- #  of gen filters in first conv layer
  ndf = 64,               -- #  of discrim filters in first conv layer
  nThreads = 4,           -- #  of data loading threads to use
  niter = 1000,             -- #  of iter at starting learning rate
  lr = 0.0002,            -- initial learning rate for adam
  lr_decay = 0.5,            -- initial learning rate for adam
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

if opt.init_g == '' then
  prep_noise = nn.Sequential()
    :add(nn.View(-1,opt.nz))
    :add(nn.Linear(opt.nz, ngf*4))
  prep_txt_g = nn.Sequential()
    :add(nn.View(-1,opt.txtSize))
    :add(nn.Linear(opt.txtSize, ngf*4))
  txtGlobalG = nn.Sequential()
    :add(nn.ParallelTable()
      :add(nn.Sequential()        -- { noise, txt }
        :add(nn.SelectTable(2))   -- txt
        :add(prep_txt_g)
        :add(nn.Replicate(16,3))
        :add(nn.Replicate(16,4))
        :add(nn.Transpose({2,3},{3,4}))) -- chw to hwc
        -- 512 x 16 x 16
      :add(nn.Sequential()        -- loc
        :add(nn.View(-1,2,3))
        :add(nn.AffineGridGeneratorBHWD(16, 16))))
    :add(nn.BilinearSamplerBHWD())
    :add(nn.Transpose({3,4},{2,3})) -- hwc to chw
    :add(nn.Contiguous())
    :add(nn.View(-1,opt.num_elt,ngf*4,16,16))
    :add(nn.Mean(2))   -- average over elements
    -- (512) x 16 x 16
    :add(SpatialConvolution(ngf*4, ngf*4, 4, 4, 2, 2, 1, 1))
    :add(SpatialBatchNormalization(ngf*4)):add(nn.ReLU(true))
    -- (ngf*2) x 8 x 8
    :add(SpatialConvolution(ngf*4, ngf*4, 4, 4, 2, 2, 1, 1))
    :add(SpatialBatchNormalization(ngf*4)):add(nn.ReLU(true))
    -- (ngf*4) x 4 x 4
    :add(SpatialConvolution(ngf*4, ngf*4, 4, 4))
    :add(SpatialBatchNormalization(ngf*4)):add(nn.ReLU(true))
    -- (ngf*4) x 1 x 1
  combiner_local = nn.Sequential()
    :add(nn.ParallelTable()
      :add(prep_noise)   -- (ngf*4)
      :add(prep_txt_g))  -- (ngf*4)
    :add(nn.JoinTable(2))
    :add(nn.ReLU(true))
  combiner_global = nn.Sequential()
    :add(nn.ConcatTable()
      :add(nn.Sequential()
        :add(nn.SelectTable(1))          -- get { noise, txt }
        :add(nn.SelectTable(1))          -- get noise
        :add(nn.Select(2,1))             -- pick first noise element
        :add(nn.Linear(opt.nz, ngf*4))   -- (ngf*4)
        :add(nn.ReLU(true)))
      :add(txtGlobalG))                  -- (ngf*4) x 1 x 1
    :add(nn.JoinTable(2))
    :add(nn.ReLU(true))
  -- global path
  globalG = nn.Sequential()
    :add(combiner_global)
    :add(nn.View(-1, ngf*8, 1, 1))
    :add(SpatialFullConvolution(ngf*8, ngf * 8, 4, 4))
    :add(SpatialBatchNormalization(ngf * 8))
    :add(nn.ReLU(true))
    -- state size: (ngf*8) x 4 x 4
    :add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
    :add(SpatialBatchNormalization(ngf * 4))
    :add(nn.ReLU(true))
    -- state size: (ngf*4) x 8 x 8
    :add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
    :add(SpatialBatchNormalization(ngf * 2))
    -- state size: (ngf*2) x 16 x 16
    :add(nn.ReLU(true))
  -- regions path
  convG = nn.Sequential()
    :add(combiner_local)
    :add(nn.View(-1, ngf*8, 1, 1))
    :add(SpatialFullConvolution(ngf*8, ngf * 8, 4, 4))
    :add(SpatialBatchNormalization(ngf * 8))
    :add(nn.ConcatTable()
      :add(nn.Sequential()
        -- state size: (ngf*8) x 4 x 4
        :add(SpatialConvolution(ngf * 8, ngf * 2, 1, 1, 1, 1, 0, 0))
        :add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
        :add(SpatialConvolution(ngf * 2, ngf * 2, 3, 3, 1, 1, 1, 1))
        :add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
        :add(SpatialConvolution(ngf * 2, ngf * 8, 3, 3, 1, 1, 1, 1))
        :add(SpatialBatchNormalization(ngf * 8)))
      :add(nn.Identity()))
    :add(nn.CAddTable())
    :add(nn.ReLU(true))
    -- state size: (ngf*8) x 4 x 4
    :add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
    :add(SpatialBatchNormalization(ngf * 4))
    -- state size: (ngf*4) x 8 x 8
    :add(nn.ConcatTable()
      :add(nn.Sequential()
        :add(SpatialConvolution(ngf * 4, ngf, 1, 1, 1, 1, 0, 0))
        :add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
        :add(SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1))
        :add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
        :add(SpatialConvolution(ngf, ngf * 4, 3, 3, 1, 1, 1, 1))
        :add(SpatialBatchNormalization(ngf * 4)))
      :add(nn.Identity()))
    :add(nn.CAddTable())
    :add(nn.ReLU(true))
    -- state size: (ngf*4) x 8 x 8
    :add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
    :add(SpatialBatchNormalization(ngf * 2))
    -- here we are at 16 x 16.
    :add(nn.Transpose({2,3},{3,4})) -- chw to hwc
  regionsG = nn.Sequential()
    :add(nn.ParallelTable()
      :add(convG)
      :add(nn.Sequential()
        :add(nn.View(-1,2,3))
        :add(nn.AffineGridGeneratorBHWD(16, 16))))
    :add(nn.BilinearSamplerBHWD())
    :add(nn.Transpose({3,4},{2,3})) -- hwc to chw
    -- unfold elements
    :add(nn.View(-1, opt.num_elt, ngf * 2, 16, 16))
    -- average over elements
    :add(nn.Mean(2))
    :add(nn.ReLU(true))
  -- combines global and local region information.
  -- inputs are assumed to be  { { noise, txt }, loc }
  netG = nn.Sequential()
    :add(nn.ConcatTable() 
      :add(globalG)
      :add(regionsG))
    :add(nn.JoinTable(2))
    -- mix them together
    :add(SpatialFullConvolution(ngf * 4, ngf * 2, 3, 3, 1, 1, 1, 1))
    :add(SpatialBatchNormalization(ngf*2)):add(nn.ReLU(true))
    -- state size: (ngf*4) x 16 x 16
    :add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
    :add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
    -- state size: (ngf) x 32 x 32
    :add(SpatialFullConvolution(ngf, 32, 4, 4, 2, 2, 1, 1))
    :add(SpatialBatchNormalization(32)):add(nn.ReLU(true))
    -- state size: (16) x 64 x 64
    :add(SpatialFullConvolution(32, nc, 4, 4, 2, 2, 1, 1))
    :add(nn.Tanh())
    -- state size: (nc) x 128 x 128
  netG:apply(weights_init)
else
  netG = torch.load(opt.init_g)
end

if opt.init_d == '' then
  -- netD expects {img, loc, txt}
  imgGlobalD = nn.Sequential()
    :add(nn.SelectTable(1))
    -- state size: (nc) x 128 x 128
    :add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
    :add(nn.LeakyReLU(0.2, true))
    -- state size: (nc) x 64 x 64
    :add(SpatialConvolution(ndf, ndf, 4, 4, 2, 2, 1, 1))
    :add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 32 x 32
    :add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
    :add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 16 x 16
    :add(SpatialConvolution(ndf * 2, ndf * 2, 3, 3, 1, 1, 1, 1))
    :add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 16 x 16
  prep_txt_d = nn.Sequential()
    :add(nn.SelectTable(3))
    :add(nn.View(-1,opt.txtSize))
    :add(nn.Linear(opt.txtSize, opt.nt))
    :add(nn.LeakyReLU(0.2,true))
  prep_loc_d = nn.Sequential()
    :add(nn.SelectTable(2))
    :add(nn.View(-1,2,3))
    :add(nn.AffineGridGeneratorBHWD(16, 16))
  -- region pathway
  imgTextRegionD = nn.Sequential()
    :add(nn.ConcatTable()
      :add(nn.Sequential()
        -- (ndf*2) x 16 x 16
        :add(imgGlobalD)
        -- opt.num_elt x (ndf*2) x 16 x 16
        :add(nn.Replicate(opt.num_elt,2))
        -- (ndf*2) x 16 x 16
        :add(nn.View(-1,ndf*2,16,16)))
      :add(nn.Sequential()       -- text path
        :add(prep_txt_d) -- opt.nt
        -- (opt.nt) x 1
        :add(nn.Replicate(opt.num_elt,2))
        -- opt.num_elt x (opt.nt) x 1
        :add(nn.View(-1,opt.nt))
        -- (opt.nt) x 1
        :add(nn.Replicate(16,3))
        -- (opt.nt) x 16
        :add(nn.Replicate(16,4))))
        -- (opt.nt) x 16 x 16
    :add(nn.JoinTable(2))
    :add(SpatialConvolution(ndf * 2 + opt.nt, ndf * 2, 3, 3, 1, 1, 1, 1))
    :add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    :add(nn.Transpose({2,3},{3,4}))
  regionD = nn.Sequential()
    :add(nn.ConcatTable() -- keypoint multiplication
      :add(imgTextRegionD)
      :add(prep_loc_d))
    :add(nn.BilinearSamplerBHWD())
    :add(nn.Transpose({3,4},{2,3}))
    -- (ndf*2) x 16 x 16
    :add(nn.Contiguous())
    -- state size: (ndf*2) x 16 x 16
    :add(SpatialConvolution(ndf * 2, ndf * 2, 1, 1))
    :add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 16 x 16
    :add(SpatialConvolution(ndf * 2, ndf, 1, 1))
    -- state size: (ndf) x 16 x 16
    :add(nn.Mean(4))
    :add(nn.Mean(3))
    :add(nn.LeakyReLU(0.2, true))
  -- global pathway
  -- state size: (ndf*2) x 16 x 16
  convGlobalD = nn.Sequential()
    :add(imgGlobalD)
    -- (ndf*2) x 16 x 16
    :add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
    :add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2,true))
    -- (ndf*4) x 8 x 8
    :add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
    :add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2,true)) -- now 4x4
    -- (ndf*8) x 4 x 4
  txtGlobalD = nn.Sequential()
    :add(nn.ConcatTable()
      :add(nn.Sequential()
        :add(prep_txt_d)          -- opt.nt
        :add(nn.Replicate(16,3))
        :add(nn.Replicate(16,4))
        :add(nn.Transpose({2,3},{3,4}))) -- chw to hwc
        -- 512 x 16 x 16
      :add(nn.Sequential()        -- loc
        :add(nn.SelectTable(2))
        :add(nn.View(-1,2,3))
        :add(nn.AffineGridGeneratorBHWD(16, 16))))
    :add(nn.BilinearSamplerBHWD())
    :add(nn.Transpose({3,4},{2,3})) -- hwc to chw
    :add(nn.Contiguous())
    :add(nn.View(-1,opt.num_elt,opt.nt,16,16))
    :add(nn.Mean(2))   -- average over elements
    -- (opt.nt) x 16 x 16
    :add(SpatialConvolution(opt.nt, ndf*4, 4, 4, 2, 2, 1, 1))
    :add(SpatialBatchNormalization(ndf*4)):add(nn.ReLU(true))
    -- (ndf) x 8 x 8
    :add(SpatialConvolution(ndf*4, ndf, 4, 4, 2, 2, 1, 1))
    :add(SpatialBatchNormalization(ndf)):add(nn.ReLU(true))
    -- (ndf*) x 4 x 4
  globalD = nn.Sequential()
    :add(nn.ConcatTable()
      :add(convGlobalD)
      :add(txtGlobalD))
    :add(nn.JoinTable(2))
    :add(nn.Contiguous())
    -- state size: (ndf*9) x 4 x 4
    :add(SpatialConvolution(ndf * 9, ndf * 4, 1, 1))
    :add(SpatialBatchNormalization(ndf * 4))
    :add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*4) x 4 x 4
    :add(SpatialConvolution(ndf * 4, ndf, 4, 4))
    :add(SpatialBatchNormalization(ndf)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 1 x 1
    :add(nn.View(-1,ndf))
  netD = nn.Sequential()
    :add(nn.ConcatTable()
      :add(regionD)
      :add(globalD))
    :add(nn.JoinTable(2))
    :add(nn.Linear(ndf * 2, ndf))
    :add(nn.BatchNormalization(ndf)):add(nn.LeakyReLU(0.2, true))
    :add(nn.Linear(ndf, 1))
    :add(nn.Sigmoid())
  netD:apply(weights_init)
else
  netD = torch.load(opt.init_d)
end

netI = nn.Sequential()
netI:add(nn.View(-1,2,3))
netI:add(nn.CropInvert())
netI:add(nn.View(-1,opt.num_elt,2,3))

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
----------------------------------------------------------------------------
local input_img = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local input_fake = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local input_dbg = torch.Tensor(opt.batchSize*opt.num_elt, 3, opt.fineSize, opt.fineSize)
local input_txt = torch.Tensor(opt.batchSize, opt.num_elt, opt.txtSize)
local input_txt_shuf = torch.Tensor(opt.batchSize, opt.num_elt, opt.txtSize)
local input_txt_flat = torch.Tensor(opt.batchSize*opt.num_elt, opt.txtSize)
local input_loc = torch.Tensor(opt.batchSize, opt.num_elt, 2, 3)
local input_loc_g = torch.Tensor(opt.batchSize, opt.num_elt, 2, 3)
local input_loc_shuf = torch.Tensor(opt.batchSize, opt.num_elt, 2, 3)
local noise = torch.Tensor(opt.batchSize, opt.num_elt, nz)
local label = torch.Tensor(opt.batchSize)
local errD, errG, errW
----------------------------------------------------------------------------
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   input_img = input_img:cuda()
   input_fake = input_fake:cuda()
   input_dbg = input_dbg:cuda()
   input_txt = input_txt:cuda()
   input_txt_shuf = input_txt_shuf:cuda()
   input_txt_flat = input_txt_flat:cuda()
   input_loc = input_loc:cuda()
   input_loc_g = input_loc_g:cuda()
   input_loc_shuf = input_loc_shuf:cuda()
   noise = noise:cuda()
   label = label:cuda()
   netD:cuda()
   netG:cuda()
   netI:cuda()
   criterion:cuda()
end

if opt.use_cudnn == 1 then
  cudnn = require('cudnn')
  netD = cudnn.convert(netD, cudnn)
  netG = cudnn.convert(netG, cudnn)
  netI = cudnn.convert(netI, cudnn)
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end

local sample = function()
  data_tm:reset(); data_tm:resume()
  real_img, real_txt, real_loc, dbg = data:getBatch()
  data_tm:stop()
  dbg = torch.reshape(dbg, opt.batchSize * opt.num_elt,
                      dbg:size(3), dbg:size(4), dbg:size(5))
  input_dbg:copy(dbg)
  input_img:copy(real_img:select(2,1))
  input_txt:copy(real_txt)
  input_loc:copy(real_loc)
  if opt.gpu >= 0 then
    input_loc:cuda()
  end
  input_loc_g:copy(netI:forward(input_loc))
  local shuf_ix = torch.randperm(input_loc:size(1))
  for n = 1,input_loc:size(1) do
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

  local fake = netG:forward({ {noise, input_txt}, input_loc_g})
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
  errW = errD_wrong

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
  local deltas = netG:backward({ { noise, input_txt }, input_loc_g}, df_dr[1])

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
    optim.adam(fDx, parametersD, optimStateD)
    optim.adam(fGx, parametersG, optimStateG)

    if opt.dbg == 1 then
      disp.image(input_dbg, {win=opt.display_id, title=opt.name})
      debug.debug()
    end

    -- logging
    if ((i-1) / opt.batchSize) % opt.print_every == 0 then
      print(('[%d][%d/%d] T:%.3f  DT:%.3f lr: %.4g '
                .. ' G:%.3f  D:%.3f W:%.3f, fs:%.2f'):format(
              epoch, ((i-1) / opt.batchSize),
              math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
              tm:time().real, data_tm:time().real,
              optimStateG.learningRate,
              errG and errG or -1, errD and errD or -1, errW and errW or -1,
              fake_score))
      local fake = netG.output
      local bbox = util.affine_to_bbox(fake:size(), input_loc:select(2,1):narrow(1,1,4))
      fake:mul(2):add(-1)
      for b = 1,bbox:size(2) do
        fake[b] = util.draw_box(fake[b], bbox[{{},b}])
      end
      disp.image(fake:narrow(1,1,4), {win=opt.display_id, title=opt.name})
      disp.image(real_img:select(2,1):narrow(1,1,4), {win=opt.display_id * 3, title=opt.name})
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

