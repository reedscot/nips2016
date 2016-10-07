
require 'image'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'lfs'
require 'stn'
util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
local dict = {}
for i = 1,#alphabet do
  dict[alphabet:sub(i,i)] = i
end
ivocab = {}
for k,v in pairs(dict) do
  ivocab[v] = k
end

opt = {
  dataset = 'cub',
  doc_length = 201,
  prefix = 'move',
  batchSize = 16,         -- number of samples to produce
  noisetype = 'normal',  -- type of noise distribution (uniform / normal).
  imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px, ...
  noisemode = 'random',  -- random / line / linefull1d / linefull
  gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
  display = 0,           -- Display image: 0 = false, 1 = true
  nz = 100,              
  net_gen = '/',
  net_txt = '',
  loadSize = 150,
  fineSize = 128,
  txtSize = 1024,
  data_root = '/mnt/brain3/datasets/txt2img/cub_ex_part',
  img_dir = '/mnt/brain1/scratch/reedscot/nx/reedscot/data/gutenbirds/CUB_200_2011/images',
  demo = 'keypoints_given',
  num_elt = 15,
  nsample = 20,
  keypoint_dim = 16,
}

dofile('data/donkey_folder_cub_keypoint.lua')

--  demo = 'trans' -- trans|stretch|shrink
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

assert(opt.net_gen ~= '')
assert(opt.net_txt ~= '')
assert(opt.net_kp ~= '')

if opt.gpu > 0 then
  ok, cunn = pcall(require, 'cunn')
  ok2, cutorch = pcall(require, 'cutorch')
  cutorch.setDevice(opt.gpu)
end

net_gen = torch.load(opt.net_gen)
net_txt = torch.load(opt.net_txt).protos.enc_doc

net_gen:evaluate()
net_txt:evaluate()

function decode(txt)
  local str = ''
  for w_ix = 1,txt:size(1) do
    local ch_ix = txt[w_ix]
    local ch = ivocab[ch_ix]
    if (ch  ~= nil) then
      str = str .. ch
    end
  end
  return str
end

local html = '<html><body><h1>Generated Images</h1><table border="1" style="width=100%"><tr><td>Caption</td><td>Image</td></tr>'

local cur_files = dir.getfiles(opt.data_root)
local ix_file = torch.randperm(#cur_files)
for n = 1,opt.nsample do
  local info = torch.load(cur_files[ix_file[n]])
  local img_file = opt.img_dir .. '/' .. info.img
  _, locs, _ = trainHook(img_file, info.parts:clone():double())

  -- prepare text
  local query = decode(info.char[{{},1}])
  print(string.format('sample %d of %d: [%s]', n, opt.nsample, query))

  -- Take unseen text embedding.
  local fea_txt = info.txt[10]:clone()
  fea_txt = torch.repeatTensor(fea_txt, opt.batchSize, 1):cuda()

  -- prepare noise
  noise = torch.Tensor(opt.batchSize, opt.nz)
  if opt.noisetype == 'uniform' then
    noise:uniform(-1, 1)
  elseif opt.noisetype == 'normal' then
    noise:normal(0, 1)
  end
  noise = noise:cuda()

  -- prepare keypoints
  local fea_loc = torch.zeros(1, opt.num_elt, opt.keypoint_dim, opt.keypoint_dim)
  for s = 1,opt.num_elt do
    local point = locs[s]
    if point[3] > 0.1 then
      local x = math.min(opt.keypoint_dim,
                  math.max(1,torch.round(point[1] * opt.keypoint_dim)))
      local y = math.min(opt.keypoint_dim,
                  math.max(1,torch.round(point[2] * opt.keypoint_dim)))
      fea_loc[{1,s,y,x}] = 1
    end
  end
  fea_loc = torch.repeatTensor(fea_loc, opt.batchSize, 1, 1, 1)
  fea_loc = fea_loc:cuda()

  local images = net_gen:forward({ { noise, fea_txt }, fea_loc }):clone()
  images:add(1):mul(0.5)
  local locs_tmp = locs:clone()
  locs_tmp:narrow(2,1,2):mul(opt.fineSize)
  images = torch.repeatTensor(images, 2, 1, 1, 1)
  for b = 1,opt.batchSize do
    images[b] = util.draw_keypoints(images[b], locs_tmp, 0.06, 1)
  end

  images = images:narrow(1,1,3)

  lfs.mkdir('results')
  local visdir = 'results/cub_kp_given'
  lfs.mkdir(visdir)
  local fname = string.format('%s/cub_%s_%d', visdir, opt.demo, n)
  local fname_png = fname .. '.png'
  image.save(fname_png, image.toDisplayTensor(images, 4, 3))

  fname = string.format('cub_kp_given/cub_%s_%d', opt.demo, n)
  local fname_rel = fname .. '.png'
  html = html .. string.format('\n<tr><td>%s</td><td><img src=\"%s\"></td></tr>',
                               query, fname_rel)
end

html = html .. '</html>'
fname_html = string.format('results/%s_%s.html', opt.dataset, opt.demo)
os.execute(string.format('echo "%s" > %s', html, fname_html))

