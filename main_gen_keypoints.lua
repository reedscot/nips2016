--[[ 

Scrip to train keypoints-generating model.

--]]
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
util = paths.dofile('util.lua')

opt = {
  num_holdout = 0,
  numCaption = 1,
  trainfiles = '',
  drop_prob = 0.9,
  port = 8000,
  dbg = 0,
  num_elt = 15,
  save_every = 10,
  print_every = 1,
  dataset = 'cub_parts',
  img_dir = '',
  filenames = '',
  data_root = '/mnt/brain3/datasets/txt2img/cub_ex_part',
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
  nThreads = 1,           -- #  of data loading threads to use
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
  ok3, cudnn = pcall(require, 'cudnn')
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

if opt.init_g == '' then
  maskOn = nn.Sequential()
    :add(nn.SelectTable(3)) -- get loc
    :add(nn.Narrow(3,3,1)) -- get presence/absence
    :add(nn.Replicate(3,3))
  maskOff = nn.Sequential()
    :add(maskOn)
    :add(nn.MulConstant(-1))
    :add(nn.AddConstant(1))
  convG = nn.Sequential()
    :add(nn.ParallelTable()
      :add(nn.Sequential() -- noise encoder
        :add(nn.Linear(opt.nz,ngf*4))
        :add(nn.ReLU(true)))
      :add(nn.Sequential() -- text encoder
        :add(nn.Linear(opt.txtSize,ngf*4))
        :add(nn.ReLU(true)))
      :add(nn.Sequential() -- loc encoder
        :add(nn.View(-1,opt.num_elt*3))
        :add(nn.Linear(opt.num_elt*3, ngf*4))
        :add(nn.BatchNormalization(ngf*4)):add(nn.ReLU(true))
        :add(nn.Linear(ngf*4,ngf*2))
        :add(nn.BatchNormalization(ngf*2)):add(nn.ReLU(true))))
    :add(nn.JoinTable(2))
    :add(nn.Linear(ngf*10,ngf*8))
    :add(nn.BatchNormalization(ngf*8)):add(nn.ReLU(true))
    :add(nn.Linear(ngf*8,ngf*4))
    :add(nn.BatchNormalization(ngf*4)):add(nn.ReLU(true))
    :add(nn.Linear(ngf*4,ngf*2))
    :add(nn.BatchNormalization(ngf*2)):add(nn.ReLU(true))
    :add(nn.Linear(ngf*2,opt.num_elt*3))
    :add(nn.View(-1,opt.num_elt,3))
    :add(nn.Sigmoid())
    
  netG = nn.Sequential()
    :add(nn.ConcatTable()
      :add(nn.Sequential()    -- generated keypoints
        :add(nn.ConcatTable()
          :add(convG)
          :add(maskOff))
        :add(nn.CMulTable()))
      :add(nn.Sequential()    -- conditioning keypoints
        :add(nn.ConcatTable()
          :add(nn.SelectTable(3)) 
          :add(maskOn))
        :add(nn.CMulTable())))
    :add(nn.CAddTable())
  netG:apply(weights_init)
else
  netG = torch.load(opt.init_g)
end

if opt.init_d == '' then
  netD = nn.Sequential()
    :add(nn.ParallelTable()
      :add(nn.Sequential() -- loc encoder
        :add(nn.View(-1,opt.num_elt*3))
        :add(nn.Linear(opt.num_elt*3, ngf*4))
        :add(nn.BatchNormalization(ndf*4)):add(nn.LeakyReLU(0.2,true))
        :add(nn.Linear(ndf*4,ndf*2))
        :add(nn.BatchNormalization(ndf*2)):add(nn.LeakyReLU(0.2,true)))
      :add(nn.Sequential() -- text encoder
        :add(nn.Linear(opt.txtSize, ndf*4))
        :add(nn.BatchNormalization(ndf*4)):add(nn.LeakyReLU(0.2,true))
        :add(nn.Linear(ndf*4,ndf*2))
        :add(nn.BatchNormalization(ndf*2)):add(nn.LeakyReLU(0.2,true))))
    :add(nn.JoinTable(2))
    :add(nn.Linear(ndf*4,ndf*2))
    :add(nn.BatchNormalization(ndf*2)):add(nn.LeakyReLU(0.2,true))
    :add(nn.Linear(ndf*2,1))
    :add(nn.Sigmoid())
  netD:apply(weights_init)
else
  netD = torch.load(opt.init_d)
end

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
local input_txt = torch.Tensor(opt.batchSize, opt.txtSize)
local input_dbg = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local input_loc = torch.Tensor(opt.batchSize, opt.num_elt, 3)
local input_loc_g = torch.Tensor(opt.batchSize, opt.num_elt, 3)
local noise = torch.Tensor(opt.batchSize, nz)
local label = torch.Tensor(opt.batchSize)
local errD, errG
----------------------------------------------------------------------------
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   input_img = input_img:cuda()
   input_txt = input_txt:cuda()
   input_fake = input_fake:cuda()
   input_dbg = input_dbg:cuda()
   input_loc = input_loc:cuda()
   input_loc_g = input_loc_g:cuda()
   noise = noise:cuda()
   label = label:cuda()
   netD:cuda()
   netG:cuda()
   criterion:cuda()
end

if (opt.gpu >= 0) and (opt.use_cudnn == 1) then
  cudnn = require('cudnn')
  netD = cudnn.convert(netD, cudnn)
  netG = cudnn.convert(netG, cudnn)
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end

local sample = function()
  data_tm:reset(); data_tm:resume()
  real_img, real_txt, real_loc, dbg = data:getBatch()
  data_tm:stop()

  --input_img:copy(real_img)
  input_txt:copy(real_txt)
  input_dbg:copy(dbg)
  input_loc:copy(real_loc)
  input_loc_g:copy(input_loc)

  -- copy a subset of input_loc into input_loc_g
  for b = 1,opt.batchSize do
    local drop = torch.rand(opt.num_elt):lt(opt.drop_prob)
    for s = 1,opt.num_elt do
      if input_loc[{b,s,3}] < 0.01 then
        input_loc[{b,s,{}}]:fill(0.0)
      end
      if drop[s] == 1 then
        input_loc_g[{b,s,{}}]:fill(0.0)
      end
    end
  end
end

-- create closure to evaluate f(X) and df/dX of discriminator
fake_score = 0.5
local fDx = function(x)
  gradParametersD:zero()

  -- train with real
  label:fill(real_label)
  local output = netD:forward{input_loc,input_txt}
  errD_real = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  local deltas = netD:backward({input_loc, input_txt}, df_do)

  -- train with fake
  if opt.noise == 'uniform' then -- regenerate random noise
    noise:uniform(-1, 1)
  elseif opt.noise == 'normal' then
    noise:normal(0, 1)
  end
  label:fill(fake_label)
  local fake = netG:forward{noise,input_txt,input_loc_g}
  input_loc:copy(fake)
  local output = netD:forward{input_loc,input_txt}
  -- update fake score tracker
  local cur_score = output:mean()
  fake_score = 0.99 * fake_score + 0.01 * cur_score
  local errD_fake = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  netD:backward({input_loc,input_txt}, df_do)

  errD = errD_real + errD_fake
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
  local df_dr = netD:updateGradInput({input_loc, input_txt}, df_do)
  local deltas = netG:backward({noise,input_txt,input_loc_g}, df_dr[1])

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
                .. ' G:%.3f  D:%.3f fs:%.2f'):format(
              epoch, ((i-1) / opt.batchSize),
              math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
              tm:time().real, data_tm:time().real,
              optimStateG.learningRate,
              errG and errG or -1, errD and errD or -1,
              fake_score))
      local fake_loc = input_loc:narrow(1,1,4):clone()
      fake_loc:narrow(3,1,2):mul(real_img:size(3))
      local fake_img = real_img:narrow(1,1,4):clone():fill(0) -- empty image
      for b = 1,fake_loc:size(1) do
        fake_img[b] = util.draw_keypoints(fake_img[b], fake_loc[b])
      end
      disp.image(fake_img, {win=opt.display_id, title=opt.name})
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

