require 'image'
dir = require 'pl.dir'
util = paths.dofile('../util.lua')

trainLoader = {}

if opt.num_holdout == nil then
  opt.num_holdout = 0
end

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

function decode(txt)
  local str = ''
  for w_ix = 1,txt:size(1) do
    local ch_ix = txt[{w_ix,1}]
    local ch = ivocab[ch_ix]
    if (ch ~= nil) then
      str = str .. ch
    end
  end
  return str
end

trainLoader.alphabet = alphabet
trainLoader.alphabet_size = alphabet_size
trainLoader.dict = dict
trainLoader.ivocab = ivocab
trainLoader.decode = decoder

local files = {}
local size = 0
cur_files = dir.getfiles(opt.data_root)
size = size + #cur_files

--------------------------------------------------------------------------------------------
local loadSize   = {3, opt.loadSize}
local sampleSize = {3, opt.fineSize}

local function loadImage(path)
  local input = image.load(path, 3, 'float')
  return input
end

local fix_boxes = function(img, locs)
  local iH = img:size(2)
  local iW = img:size(3)
  for b = 1,locs:size(1) do
    local x = locs[{b,1}]
    local y = locs[{b,2}]
    local present = locs[{b,3}]
    locs[{1,b}] = x / iW
    locs[{2,b}] = y / iH
  end
  return locs
end

-- function to load the image, jitter it appropriately (random crops etc.)
trainHook = function(path, loc)
  collectgarbage()
  local input = loadImage(path)
  local iW_orig = input:size(3)
  local iH_orig = input:size(2)

  -- first scale to load size.
  input = image.scale(input, loadSize[2], loadSize[2])
  local iW = input:size(3)
  local iH = input:size(2)

  -- rescale bboxes
  local x_scale_factor = iW / iW_orig
  local y_scale_factor = iH / iH_orig
  for b = 1,loc:size(1) do
    loc[{b,1}] = math.max(1, math.floor(x_scale_factor * loc[{b,1}]))
    loc[{b,2}] = math.max(1, math.floor(y_scale_factor * loc[{b,2}]))
  end

  -- then crop and flip
  local oW = sampleSize[2]
  local oH = sampleSize[2]
  local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
  local w1 = math.ceil(torch.uniform(1e-2, iW-oW))

  local out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
  assert(out:size(2) == oW)
  assert(out:size(3) == oH)

  local do_flip = torch.uniform() > 0.5
  if do_flip then
    out = image.hflip(out);
  end
  
  -- update bboxes
  for b = 1,loc:size(1) do
    loc[{b,1}] = math.max(1, loc[{b,1}] - w1) -- x
    loc[{b,2}] = math.max(1, loc[{b,2}] - h1) -- y
    if do_flip then
      loc[{b,1}] = math.max(1, oW - loc[{b,1}] + 1)
    end
  end

  input_dbg = util.draw_keypoints(out:clone(), loc)

  -- then scale to final sampleSize
  local out = image.scale(out, sampleSize[2], sampleSize[2])
  out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]

  -- compute affine transformation matrices.
  for b = 1,loc:size(1) do
    local bW =  loc[{b,1}]
    local bH =  loc[{b,2}]
    local scale_X = bW / out:size(3)
    local scale_Y = bH / out:size(2)

    loc[{b,1}] = scale_X
    loc[{b,2}] = scale_Y
  end

  return out, loc, input_dbg
end

function trainLoader:sample(quantity)
  local ix_file = torch.Tensor(quantity)
  for n = 1, quantity do
    local samples = torch.randperm(#cur_files):narrow(1,1,2)
    local file_ix = samples[1]
    ix_file[n] = file_ix
  end

  local data_img = torch.zeros(quantity, sampleSize[1], sampleSize[2], sampleSize[2])
  local data_dbg = torch.zeros(quantity, sampleSize[1], sampleSize[2], sampleSize[2])
  local data_txt = torch.zeros(quantity, opt.txtSize)
  local data_loc = torch.zeros(quantity, opt.num_elt, opt.keypoint_dim, opt.keypoint_dim)
  local locs_raw = torch.zeros(quantity, opt.num_elt, 3)

  for n = 1, quantity do
    local t7file = cur_files[ix_file[n]]
    local info = torch.load(t7file)

    local img_file = opt.img_dir .. '/' .. info.img

    -- normalize image and keypoint locations.
    local img, locs, img_dbg = trainHook(img_file, info.parts:clone():double())
    --locs_raw[n]:copy(locs)
    locs_raw[n]:copy(locs:narrow(1,1,opt.num_elt))
    data_img[{n,{},{},{}}]:copy(img)
    data_dbg[{n,{},{},{}}]:copy(img_dbg)

    -- add text information.
    local ix_txt = torch.randperm(info.txt:size(1) - opt.num_holdout)
    for c = 1, opt.numCaption do
      data_txt[{n,{}}]:add(info.txt[ix_txt[c] ]:double())
    end
    data_txt[{n,{}}]:div(opt.numCaption)

    for s = 1,opt.num_elt do
      local point = locs[s]
      if point[3] > 0.1 then
        local x = math.min(opt.keypoint_dim,
                    math.max(1,torch.round(point[1] * opt.keypoint_dim)))
        local y = math.min(opt.keypoint_dim,
                    math.max(1,torch.round(point[2] * opt.keypoint_dim)))
        data_loc[{n,s,y,x}] = 1
      end
    end
  end
  collectgarbage(); collectgarbage()
  return data_img, data_txt, data_loc, data_dbg, locs_raw
end

function trainLoader:size()
  return size
end

