require 'image'
dir = require 'pl.dir'

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
  for b = 1,locs:size(2) do
    local x = locs[{1,b}]
    local y = locs[{2,b}]
    local width = locs[{3,b}]
    local height = locs[{4,b}]

    locs[{1,b}] = math.max(1,x)
    locs[{2,b}] = math.max(1,y)
    locs[{3,b}] = locs[{3,b}] - math.max(0, x + width - iW)
    locs[{4,b}] = locs[{4,b}] - math.max(0, y + height - iH)
  end
  return locs
end

local draw_boxes = function(img, locs, thickness)
  thickness = thickness or 3
  local imgs = {}
  local imgs_mat = torch.zeros(locs:size(2), 3, sampleSize[2], sampleSize[2])
  local red = torch.zeros(3)
  red[2] = opt.num_elt * 1.0
  for b = 1,locs:size(2) do
    local x = locs[{1,b}]
    local y = locs[{2,b}]
    local width = locs[{3,b}]
    local height = locs[{4,b}]


    local img_tmp = img:clone()

    -- top bar
    for n = 1,width do
      for t = 1,thickness do
        local offset = -math.floor(thickness / 2) + (t-1)
        if ((y + offset) <= img_tmp:size(2) and  (y + offset) >= 1) then
          img_tmp[{{},y + offset, x+n-1}]:copy(red)
        end
      end
    end
    -- bottom bar
    for n = 1,width do
      for t = 1,thickness do
        local offset = -math.floor(thickness / 2) + (t-1)
        if ((y + height + offset) <= img_tmp:size(2) and (y + height + offset) >= 1) then
          img_tmp[{{},y + height + offset, x+n-1}]:copy(red)
        end
      end
    end
    -- left bar
    for n = 1,height do
      for t = 1,thickness do
        local offset = -math.floor(thickness / 2) + (t-1)
        if ((x + offset) <= img_tmp:size(3) and (x + offset) >= 1) then
          img_tmp[{{},y+n-1, x + offset}]:copy(red)
        end
      end
    end
    -- right bar
    for n = 1,height do
      for t = 1,thickness do
        local offset = -math.floor(thickness / 2) + (t-1)
        if ((x + width + offset) <= img_tmp:size(3) and (x + width + offset) >= 1) then
          img_tmp[{{},y+n-1, x + width + offset}]:copy(red)
        end
      end
    end

    img_tmp = image.scale(img_tmp, sampleSize[2], sampleSize[2])
    imgs_mat[b]:copy(img_tmp)
  end

  return imgs_mat
end

-- function to load the image, jitter it appropriately (random crops etc.)
trainHook = function(path, loc)
  collectgarbage()
  local input = loadImage(path)
  local iW_orig = input:size(3)
  local iH_orig = input:size(2)
  local loc_affine = torch.zeros(loc:size(2), 2, 3)

  -- handle edge cases
  local loc = fix_boxes(input, loc)

  -- first scale to load size.
  input = image.scale(input, loadSize[2], loadSize[2])
  local iW = input:size(3)
  local iH = input:size(2)

  -- rescale bboxes
  local x_scale_factor = iW / iW_orig
  local y_scale_factor = iH / iH_orig
  for b = 1,loc:size(2) do
    loc[{1,b}] = math.max(1,math.floor(x_scale_factor * loc[{1,b}])) -- x
    loc[{2,b}] = math.max(1,math.floor(y_scale_factor * loc[{2,b}])) -- y
    loc[{3,b}] = math.floor(x_scale_factor * loc[{3,b}]) -- width
    loc[{4,b}] = math.floor(y_scale_factor * loc[{4,b}]) -- height
  end

  --local input_dbg = draw_boxes(input:clone(), loc)

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
  for b = 1,loc:size(2) do
    loc[{1,b}] = math.max(1, loc[{1,b}] - w1) -- x
    loc[{2,b}] = math.max(1, loc[{2,b}] - h1) -- y
    loc[{3,b}] = loc[{3,b}] - math.max(0, loc[{1,b}] + loc[{3,b}] - oW)
    loc[{4,b}] = loc[{4,b}] - math.max(0, loc[{2,b}] + loc[{4,b}] - oH)

    if do_flip then
      loc[{1,b}] = math.max(1, oW - (loc[{1,b}] + loc[{3,b}]))
    end
  end

  local input_dbg = draw_boxes(out:clone(), loc)

  -- then scale to final sampleSize
  local out = image.scale(out, sampleSize[2], sampleSize[2])
  out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]

  -- compute affine transformation matrices.
  for b = 1,loc:size(2) do
    local b_x = loc[{1,b}]
    local b_y = loc[{2,b}]
    local bW =  loc[{3,b}]
    local bH =  loc[{4,b}]
    local t_x = 2*((b_x + bW/2.0) / out:size(3) - 0.5)
    local t_y = 2*((b_y + bH/2.0) / out:size(2) - 0.5)
    local scale_X = bW / out:size(3)
    local scale_Y = bH / out:size(2)

    loc_affine[{b,1,1}] = scale_Y
    loc_affine[{b,2,2}] = scale_X
    loc_affine[{b,1,3}] = t_y
    loc_affine[{b,2,3}] = t_x
  end

  return out, loc_affine, input_dbg
end


function trainLoader:sample(quantity)
  local ix_file = torch.Tensor(quantity)
  for n = 1, quantity do
    local samples = torch.randperm(#cur_files):narrow(1,1,2)
    local file_ix = samples[1]
    ix_file[n] = file_ix
  end

  local data_img = torch.zeros(quantity, opt.num_elt, sampleSize[1],
                               sampleSize[2], sampleSize[2])
  local data_dbg = torch.zeros(quantity, opt.num_elt, sampleSize[1],
                               sampleSize[2], sampleSize[2])
  local data_txt = torch.zeros(quantity, opt.num_elt, opt.txtSize)
  local data_loc = torch.zeros(quantity, opt.num_elt, 2, 3)

  assert(data_img ~= nil)

  for n = 1, quantity do
    local t7file = cur_files[ix_file[n]]
    local info = torch.load(t7file)
    local img_file = opt.img_dir .. '/' .. info.img

    -- normalize image and convert bounding box coordinates to affine.
    local img, locs, img_dbg = trainHook(img_file, info.loc:clone())

    local ix_txt = torch.randperm(info.txt:size(1) - opt.num_holdout)
 
    for s = 1, opt.num_elt do
      data_loc[{n,s,{},{}}]:copy(locs[1])
      data_img[{n,s,{},{},{}}]:copy(img)
      data_dbg[{n,s,{},{},{}}]:copy(img_dbg[1])
      for c = 1, opt.numCaption do
        data_txt[{n,s,{}}]:add(info.txt[ix_txt[c]]:double())
      end
      data_txt[{n,s,{}}]:div(opt.numCaption)
    end
  end
  collectgarbage(); collectgarbage()
  return data_img, data_txt, data_loc, data_dbg
end

function trainLoader:size()
  return size
end

