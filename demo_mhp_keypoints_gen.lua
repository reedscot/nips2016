
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
  trainfiles = 'mhp_trainfiles_a5.t7',
  dataset = 'mhp',
  doc_length = 201,
  batchSize = 16,         -- number of samples to produce
  noisetype = 'normal',  -- type of noise distribution (uniform / normal).
  imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px, ...
  noisemode = 'random',  -- random / line / linefull1d / linefull
  gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
  display = 0,           -- Display image: 0 = false, 1 = true
  nz = 100,              
  net_kp = '',
  net_gen = '',
  net_txt = '',
  loadSize = 140,
  fineSize = 128,
  txtSize = 1024,
  data_root = '/mnt/brain3/datasets/txt2img/mhp/t7files_loc_txt_ex',
  img_dir = '/mnt/brain3/datasets/txt2img/mhp/images',
  demo = 'keypoints_gen',
  num_elt = 17,
  nsample = 50,
  keypoint_dim = 16,
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

dofile('data/donkey_folder_mhp_keypoint_and_image.lua')

assert(opt.net_gen ~= '')
assert(opt.net_txt ~= '')
assert(opt.net_kp ~= '')

net_gen = torch.load(opt.net_gen)
net_txt = torch.load(opt.net_txt)
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

if opt.trainfiles == '' then
  cur_files = dir.getfiles(opt.data_root)
else
  cur_files = torch.load(opt.trainfiles)
  for k,v in pairs(cur_files) do
    cur_files[k] = opt.data_root .. '/' .. cur_files[k]
  end
end
local batch_info = {}
local key_word = { 'ski', 'yoga', 'golf', 'swim' }
while #batch_info < opt.nsample do
  local ix_file = torch.randperm(#cur_files)[1]
  local info = torch.load(cur_files[ix_file])
  local cap = decode(info.char[{{},1}])

  if (info.has_kp[{1,1}] == 0) then
    found = true
  end
  if found == true then
    batch_info[#batch_info + 1] = info
  end
end

local html = '<html><body><h1>Generated Images</h1><table border="1" style="width=100%"><tr><td>Caption</td><td>Image</td></tr>'

for n = 1,opt.nsample do
  local info = batch_info[n]
  local img_file = opt.img_dir .. '/' .. info.img
  local img = image.load(img_file)
  img = image.scale(img, 256, 256)

  -- prepare text
  local txt_ix = info.txt:size(1)
  local query = decode(info.char[{{},txt_ix}])
  local fea_txt = info.txt[txt_ix]:clone()
  print(string.format('sample %d of %d: [%s]', n, opt.nsample, query))

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
  local fea_loc_inp = torch.zeros(opt.batchSize, opt.num_elt,3)
  fea_loc_inp = fea_loc_inp:cuda()
  fea_loc = net_kp:forward{noise, fea_txt, fea_loc_inp}:clone()

  local data_loc = torch.zeros(opt.batchSize, opt.num_elt,
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

  local images = net_gen:forward({ { noise, fea_txt }, data_loc }):clone()
  images:add(1):mul(0.5)

  local locs_tmp = fea_loc:clone()
  locs_tmp:narrow(3,1,2):mul(opt.fineSize)

  images = torch.repeatTensor(images, 2, 1, 1, 1)
  for b = 1,opt.batchSize do
    images[b] = util.draw_keypoints(images[b], locs_tmp[b])
  end

  images = images:narrow(1,1,3)

  lfs.mkdir('results')
  local visdir = 'results/mhp_kpgen'
  lfs.mkdir(visdir)
  local fname = string.format('%s/mhp_%s_%d', visdir, opt.demo, n)
  local fname_png = fname .. '.png'
  image.save(fname_png, image.toDisplayTensor(images, 4, 3))

  fname = string.format('mhp_kpgen/mhp_%s_%d', opt.demo, n)
  local fname_rel = fname .. '.png'
  html = html .. string.format('\n<tr><td>%s</td><td><img src=\"%s\"></td></tr>',
                               query, fname_rel)
end

html = html .. '</html>'
fname_html = string.format('results/%s_%s.html', opt.dataset, opt.demo)
os.execute(string.format('echo "%s" > %s', html, fname_html))

