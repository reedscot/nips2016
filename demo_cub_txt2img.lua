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
  name = '',
  zero_kp = 0,
  gen_kp = 1,
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
  net_gen = '/home/reedscot/checkpoints/cub_part_stn_kd16_bs16_ngf128_ndf128_600_net_G.t7',
  net_txt = '../multimodal/cv/lm_sje_nc4_cub_hybrid_gru18_a1_c512_0.00070_1_10_trainvalids.txt_iter30000.t7',
  net_kp = '/home/reedscot/checkpoints/cub_kptxt2kp_large_bs64_ngf128_ndf128_200_net_G.t7',
  loadSize = 150,
  fineSize = 128,
  txtSize = 1024,
  txt = 'this is a bright red bird',
  demo = 'kptxt2kp',
  num_elt = 15,
  keypoint_dim = 16,
  data_root = '/mnt/brain3/datasets/txt2img/cub_ex_part',
  img_dir = '/mnt/brain1/scratch/reedscot/nx/reedscot/data/gutenbirds/CUB_200_2011/images',
  t7file = '',
}

dofile('data/donkey_folder_cub_keypoint.lua')

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

net_gen = torch.load(opt.net_gen)
net_txt = torch.load(opt.net_txt).protos.enc_doc
net_kp  = torch.load(opt.net_kp)

net_gen:evaluate()
net_txt:evaluate()
net_kp:evaluate()

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

-- prepare noise
noise = torch.Tensor(opt.batchSize, opt.nz)
if opt.noisetype == 'uniform' then
  noise:uniform(-1, 1)
elseif opt.noisetype == 'normal' then
  noise:normal(0, 1)
end
noise = noise:cuda()

-- prepare text.
local query_str = string.lower(opt.txt)
local txt_mat = torch.zeros(1,opt.doc_length,#alphabet)
for i = 1,#query_str do
  local ch = query_str:sub(i,i)
  local on_ix = dict[ch]
  if (on_ix ~= 0) and (on_ix ~= nil) then
    txt_mat[{1, i, on_ix}] = 1
  end
end
txt_mat = txt_mat:float():cuda()
local fea_txt = net_txt:forward(txt_mat)
fea_txt = torch.repeatTensor(fea_txt, opt.batchSize, 1)

local data_loc
local locs_raw
if opt.gen_kp == 1 then
  -- prepare keypoints
  local fea_loc_inp = torch.zeros(opt.batchSize, opt.num_elt, 3)
  fea_loc_inp = fea_loc_inp:cuda()
  locs = net_kp:forward{noise, fea_txt, fea_loc_inp}:clone()
  fea_loc = locs:clone()

  if opt.zero_kp == 1 then
    fea_loc:fill(0)
  end

  data_loc = torch.zeros(opt.batchSize, opt.num_elt,
                         opt.keypoint_dim, opt.keypoint_dim)
  for b = 1,opt.batchSize do
    for s = 1,opt.num_elt do
      local point = fea_loc[{b,s,{}}]
      if point[3] > 0.5 then
        local x = math.min(opt.keypoint_dim,
                    math.max(1,torch.round(point[1] * opt.keypoint_dim)))
        local y = math.min(opt.keypoint_dim,
                    math.max(1,torch.round(point[2] * opt.keypoint_dim)))
        data_loc[{b,s,y,x}] = 1
      end
    end
  end
  data_loc = data_loc:cuda()
else
  -- load keypoints from file.
  local info = torch.load(opt.data_root .. '/' .. opt.t7file)
  local img_file = opt.img_dir .. '/' .. info.img
  img, locs, _  = trainHook(img_file, info.parts:clone():double())
  img:add(1):mul(0.5)
  image.save('tmp_bird.png', img)

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
  data_loc = fea_loc
  locs = locs:view(1,opt.num_elt,3)
  locs = torch.repeatTensor(locs, opt.batchSize, 1, 1)
end

local images = net_gen:forward({ { noise, fea_txt }, data_loc }):clone()
images:add(1):mul(0.5)

locs_tmp = locs:clone()
locs_tmp:narrow(3,1,2):mul(opt.fineSize)

images = torch.repeatTensor(images, 2, 1, 1, 1)
for b = 1,opt.batchSize do
  images[b] = util.draw_keypoints(images[b], locs_tmp[b])
end

--local visdir = 'results_cub_paper'
local visdir = 'results_cub_supp'
lfs.mkdir(visdir)
local fname = string.format('%s/cub_g%d_%s_%s', visdir, opt.gen_kp, opt.demo, opt.name)
local fname_png = fname .. '.png'
local fname_txt = fname .. '.txt'
image.save(fname_png, image.toDisplayTensor(images, 0, opt.batchSize))
html = html .. string.format('\n<tr><td>%s</td><td><img src=\"%s\"></td></tr>',
                             query_str, fname_png)
os.execute(string.format('echo "%s" > %s', opt.txt, fname_txt))

html = html .. '</html>'
fname_html = string.format('%s_%s_%s.html', opt.dataset, opt.demo, opt.name)
os.execute(string.format('echo "%s" > %s', html, fname_html))

