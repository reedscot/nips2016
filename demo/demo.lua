local orbiter = require 'orbiter'

local hello = orbiter.new()
local mime = require("mime")
local util = paths.dofile('../util.lua')
require('image')
require('nn')
require('nngraph')
require('image')
require('json')

-- Fill in appropriate data path variable
if arg[1] == nil then
  print('Please enter path to folder containing data')
else
  DATA_PATH = arg[1]
end

net_txt = torch.load(DATA_PATH .. 'lm_sje_nc4_cub_hybrid_gru18_a1_c512_0.00070_1_10_trainvalids.txt_iter30000.t7_cpu.t7')
net_txt:evaluate()
net_gen = torch.load(DATA_PATH .. 'cub_part_stn_kd16_bs16_ngf128_ndf128_600_net_G.t7_cpu.t7')
net_gen:evaluate()
net_kp = torch.load(DATA_PATH .. 'cub_kptxt2kp_large_bs64_ngf128_ndf128_200_net_G.t7_cpu.t7')
net_kp:evaluate()

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

opt = {}
opt.keypoint_dim = 16
opt.num_elt = 15
opt.doc_length = 201
opt.batchSize = 1
opt.nz = 100
opt.txtSize = 1024
opt.noisetype = 'normal'

function hello:index(web)
  local f = assert(io.open('index.html', 'r'))
  local rtrn = f:read('*all')
  f:close()
  return rtrn
end

function hello:request(web)
  local data = json.decode(web.POST.data)
  local desc = data.description
  local showkps = data.showkps
  -- prepare noise
  noise = torch.Tensor(opt.batchSize, opt.nz)
  if opt.noisetype == 'uniform' then
    noise:uniform(-1, 1)
  elseif opt.noisetype == 'normal' then
    noise:normal(0, 1)
  end
  -- prepare text
  local txt_mat = torch.zeros(1,opt.doc_length,#alphabet)
  for t = 1,opt.doc_length do
    local ch = desc:sub(t,t)
    local on_ix = dict[ch]
    if (on_ix ~= 0 and on_ix ~= nil) then
      txt_mat[{1, t, on_ix}] = 1
    end
  end
  local fea_txt = net_txt:forward(txt_mat):clone()
  -- prepare keypoints
  local fea_loc_inp = torch.zeros(opt.batchSize, opt.num_elt, 3)
  for n = 1,#data.keypoints do
    local id = data.keypoints[n].part_id
    local x = data.keypoints[n].x / 256.0
    local y = data.keypoints[n].y / 256.0
    fea_loc_inp[{1,id,1}] = x
    fea_loc_inp[{1,id,2}] = y
    fea_loc_inp[{1,id,3}] = 1.0
  end
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
  local images = net_gen:forward({ { noise, fea_txt }, data_loc }):clone()
  images:add(1):mul(0.5)
  local img = images:select(1,1)

  local locs_tmp = fea_loc:clone()
  locs_tmp:narrow(3,1,2):mul(128)

  if showkps==1 then
    print(showkps)
    print('drawing keypoints...')
    img = util.draw_keypoints(img, locs_tmp[1], 0.03)
  end

  print(desc)

  local tmp_fname = '/tmp/tmp_bbox.jpeg'
  image.save(tmp_fname, img)
  local f = assert(io.open(tmp_fname, "rb"))
  local img_binary = f:read("*all")
  local img_b64 = 'data:image/jpeg;base64,' .. mime.b64(img_binary)
  return img_b64,'image/jpeg'
end

hello:dispatch_get(hello.index,'/','/index')
hello:dispatch_post(hello.request, '/request')
hello:dispatch_static '/resources/images/.+'
hello:dispatch_static '/resources/css/.+'
hello:dispatch_static '/resources/javascript/.+'

hello:run(...)

