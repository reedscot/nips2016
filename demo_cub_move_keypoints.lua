
image = require('image')
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
  keypoint_dim = 16,
  dataset = 'cub',
  doc_length = 201,
  prefix = 'move',
  batchSize = 1,         -- number of samples to produce
  noisetype = 'normal',  -- type of noise distribution (uniform / normal).
  imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px, ...
  noisemode = 'random',  -- random / line / linefull1d / linefull
  gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
  display = 0,           -- Display image: 0 = false, 1 = true
  nz = 100,              
  net_gen = '',
  net_txt = '',
  net_kp = '',
  num_elt = 15,
  fineSize = 128,
  loadSize = 150,
  txtSize = 1024,
  txt_file = 'cub_captions_move.txt',
  demo = 'shrink' -- trans|stretch|shrink
}
--  demo = 'trans' -- trans|stretch|shrink
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

assert(opt.net_gen ~= '')
assert(opt.net_txt ~= '')

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

local num_loc
local sx, sx_step
local sy, sy_step
local x, x_step
local y, y_step
if opt.demo == 'trans' then
  num_loc = 3
  sx      = 2
  sx_step = 0
  sy      = 2
  sy_step = 0
  y       = 0
  y_step  = 0
  x       = -0.7
  x_step  = 0.7
elseif opt.demo == 'stretch' then
  num_loc = 3
  sx      = 2
  sx_step = -0.5
  sy      = 2
  sy_step = 0
  y       = 0
  y_step  = 0.0
  x       = 0
  x_step  = 0
elseif opt.demo == 'shrink' then
  num_loc = 3
  sx      = 1.25 
  sx_step = 0.5
  sy      = 1.25
  sy_step = 0.5
  y       = 0
  y_step  = 0.0
  x       = 0
  x_step  = 0
else
  assert(false)
end

local fea_loc = torch.zeros(num_loc, 1, 2, 3)
local bbox = torch.zeros(4, num_loc)
local sz = torch.zeros(3)
sz[1] = 3
sz[2] = 128
sz[3] = 128
for n = 1,num_loc do
  local xpos = x + (n-1)*x_step
  local ypos = y + (n-1)*y_step
  local tx = xpos
  local ty = ypos

  local cur_sx = sx + (n-1)*sx_step
  local cur_sy = sy + (n-1)*sy_step

  -- local box
  fea_loc[{n, 1, 1, 1}] = cur_sx
  fea_loc[{n, 1, 2, 2}] = cur_sy
  fea_loc[{n, 1, 1, 3}] = tx
  fea_loc[{n, 1, 2, 3}] = ty

  local tmp_loc = util.invert_affine(fea_loc[n]:clone())

  bbox[{{},n}]:copy(util.affine_to_bbox(sz, tmp_loc)[{{},1}])
end
fea_loc = fea_loc:cuda()

local html = '<html><body><h1>Generated Images</h1><table border="1" style="width=100%"><tr><td>Caption</td><td>Image</td></tr>'

BEAK_IX = 2
TAIL_IX = 14

local count = 1
for query in io.lines(opt.txt_file) do
  -- encode text.
  query_str = string.lower(query)
  local txt_mat = torch.zeros(1,opt.doc_length,#alphabet)
  for i = 1,#query_str do
    if i > txt_mat:size(2) then
      break
    end
    local ch = query_str:sub(i,i)
    local on_ix = dict[ch]
    if on_ix ~= 0 then
      txt_mat[{1, i, on_ix}] = 1
    end
  end
  txt_mat = txt_mat:float():cuda()
  local fea_txt = net_txt:forward(txt_mat):clone()
  fea_txt = torch.repeatTensor(fea_txt, num_loc, 1)

  noise = torch.Tensor(1, opt.nz)
  if opt.noisetype == 'uniform' then
    noise:uniform(-1, 1)
  elseif opt.noisetype == 'normal' then
    noise:normal(0, 1)
  end
  noise = torch.repeatTensor(noise, num_loc, 1)
  noise = noise:cuda()

  -- prepare keypoints
  local kp_inp = torch.zeros(num_loc, opt.num_elt, 3)
  for n = 1,num_loc do
    kp_inp[{n, BEAK_IX, 1}] = bbox[{1,n}]
    kp_inp[{n, BEAK_IX, 2}] = bbox[{2,n}]
    kp_inp[{n, BEAK_IX, 3}] = 1.0

    kp_inp[{n, TAIL_IX, 1}] = bbox[{1,n}] + bbox[{3,n}]
    kp_inp[{n, TAIL_IX, 2}] = bbox[{2,n}] + bbox[{4,n}]
    kp_inp[{n, TAIL_IX, 3}] = 1.0
  end
  kp_inp = kp_inp:cuda()
  kp_out = net_kp:forward{noise, fea_txt, kp_inp}:clone()

  local data_loc = torch.zeros(num_loc, opt.num_elt,
                               opt.keypoint_dim, opt.keypoint_dim)
  for b = 1,num_loc do
    for s = 1,opt.num_elt do
      local point = kp_out[{b,s,{}}]
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
  for n = 1,images:size(1) do
    local beak_box = torch.zeros(4)
    beak_box[1] = kp_inp[{n, BEAK_IX, 1}] - 0.04
    beak_box[2] = kp_inp[{n, BEAK_IX, 2}] - 0.04
    beak_box[3] = 0.1
    beak_box[4] = 0.1
    local tail_box = torch.zeros(4)
    tail_box[1] = kp_inp[{n, TAIL_IX, 1}] - 0.04
    tail_box[2] = kp_inp[{n, TAIL_IX, 2}] - 0.04
    tail_box[3] = 0.1
    tail_box[4] = 0.1
    images[n] = util.draw_box(images[n], beak_box, 2)
    images[n] = util.draw_box(images[n], tail_box, 2)
  end

  lfs.mkdir('results')
  local visdir = 'results/cub_' .. opt.demo .. '_kp'
  lfs.mkdir(visdir)
  local fname = string.format('%s/cub_%s_%d', visdir, opt.demo, count)
  local fname_png = fname .. '.png'
  image.save(fname_png, image.toDisplayTensor(images, 4, num_loc))

  local vispath = 'cub_' .. opt.demo .. '_kp'
  local fname_rel = string.format('%s/cub_%s_%d', vispath, opt.demo, count)
  fname_rel = fname_rel .. '.png'
  html = html .. string.format('\n<tr><td>%s</td><td><img src="%s"></td></tr>',
                               query, fname_rel)
  count = count + 1
end

html = html .. '</html>'
fname_html = string.format('results/%s_%s_kp.html', opt.dataset, opt.demo)
os.execute(string.format('echo "%s" > %s', html, fname_html))

