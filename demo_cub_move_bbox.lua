
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
  batchSize = 1,         -- number of samples to produce
  noisetype = 'normal',  -- type of noise distribution (uniform / normal).
  imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px, ...
  noisemode = 'random',  -- random / line / linefull1d / linefull
  gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
  display = 0,           -- Display image: 0 = false, 1 = true
  nz = 100,              
  net_gen = '',
  net_txt = '',
  num_elt = 1,
  fineSize = 128,
  txtSize = 1024,
  txt_file = 'cub_captions_move.txt',
  demo = 'shrink' -- trans|stretch|shrink
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

assert(opt.net_gen ~= '')
assert(opt.net_txt ~= '')

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

local count = 1
for query in io.lines(opt.txt_file) do
  -- encode text.
  query_str = string.lower(query)
  print(query_str)
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
  fea_txt = fea_txt:reshape(1, 1, opt.txtSize)
  fea_txt = torch.repeatTensor(fea_txt, num_loc, 1, 1)

  noise = torch.Tensor(1, opt.num_elt, opt.nz)
  if opt.noisetype == 'uniform' then
    noise:uniform(-1, 1)
  elseif opt.noisetype == 'normal' then
    noise:normal(0, 1)
  end
  noise = torch.repeatTensor(noise, num_loc, 1, 1)
  noise = noise:cuda()

  local images = net_gen:forward({ { noise, fea_txt }, fea_loc }):clone()
  images = images:narrow(1,1,3)
  images:add(1):mul(0.5)
  for n = 1,images:size(1) do
    images[n] = util.draw_box(images[n], bbox[{{},n}], 3)
  end

  lfs.mkdir('results')
  local visdir = 'results/cub_' .. opt.demo .. '_bbox'
  lfs.mkdir(visdir)
  local fname = string.format('%s/cub_%s_%d', visdir, opt.demo, count)
  local fname_png = fname .. '.png'
  image.save(fname_png, image.toDisplayTensor(images, 4, num_loc))

  local vispath = 'cub_' .. opt.demo .. '_bbox'
  local fname_rel = string.format('%s/cub_%s_%d', vispath, opt.demo, count)
  fname_rel = fname_rel .. '.png'
  html = html .. string.format('\n<tr><td>%s</td><td><img src="%s"></td></tr>',
                               query, fname_rel)
  count = count + 1
end

html = html .. '</html>'
fname_html = string.format('results/%s_%s_bbox.html', opt.dataset, opt.demo)
os.execute(string.format('echo "%s" > %s', html, fname_html))

