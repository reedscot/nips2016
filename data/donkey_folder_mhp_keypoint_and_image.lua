
require 'image'
dir = require 'pl.dir'
util = paths.dofile('../util.lua')

local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "

trainLoader = {}

if opt.num_holdout == nil then
  opt.num_holdout = 0
end

if opt.trainfiles == '' then
  cur_files = dir.getfiles(opt.data_root)
else
  cur_files = torch.load(opt.trainfiles)
  for f = 1,#cur_files do
    cur_files[f] = opt.data_root .. '/' .. cur_files[f]
  end
end

act2ix = {
  ['skiing, climbing up'] = 1,
  ['skiing, cross-country'] = 1,
  ['skiing, cross-country, biathlon, skating technique'] = 1,
  ['skiing, downhill'] = 1,
  ['skiing, general'] = 1,
  ['swimming, backstroke'] = 2,
  ['swimming, breaststroke, recreational'] = 2,
  ['swimming, butterfly, general'] = 2,
  ['swimming, general'] = 2,
  ['swimming, lake, ocean, river (Taylor Codes 280, 295)'] = 2,
  ['swimming, sidestroke, general'] = 2,
  ['tennis'] = 3,
  ['tennis, doubles'] = 3,
  ['tennis, hitting balls, non-game play, moderate effort'] = 3,
  ['frisbee'] = 4,
  ['golf'] = 5,
  ['sitting, teaching stretching or yoga, or light effort exercise class'] = 6,
  ['yoga, Nadisodhana'] = 6,
  ['yoga, Power'] = 6,
}
ix2cls = {
  'bicycling',
  'conditioning exercise',
  'dancing',
  'fishing and hunting',
  'home activities',
  'home repair',
  'inactivity quiet/light',
  'lawn and garden',
  'miscellaneous',
  'music playing',
  'occupation',
  'religious activities',
  'running',
  'self care',
  'sports',
  'transportation',
  'volunteer activities',
  'walking',
  'water activities',
  'winter activities'
}
cls2ix = {}
for k,v in pairs(ix2cls) do
  cls2ix[v] = k
end

size = #cur_files

--------------------------------------------------------------------------------------------
local loadSize   = {3, opt.loadSize}
local sampleSize = {3, opt.fineSize}

local function loadImage(path)
  local input = image.load(path, 3, 'float')
  return input
end

-- function to load the image, jitter it appropriately (random crops etc.)
trainHook = function(path, points, pos, scale, head)
  collectgarbage()
  pt_head = torch.zeros(1,3)
  pt_head[{1,1}] = (head[1] + head[3]) / 2.0
  pt_head[{1,2}] = (head[2] + head[4]) / 2.0
  if head:sum() > 1e-3 then pt_head[{1,3}] = 1.0 end

  local input = loadImage(path)
  local loc = points:transpose(1,2) -- #points x 3
  loc = torch.cat(loc, pt_head, 1)

  -- First crop single person
  local oW = math.ceil(scale[1] * 200 * 1.1)
  local oH = math.ceil(scale[1] * 200 * 1.1)
  local w1 = math.max(1,math.floor(pos[1] - oW/2))
  local h1 = math.max(1,math.floor(pos[2] - oH/2))
  local w2 = math.min(input:size(3), w1 + oW - 1)
  local h2 = math.min(input:size(2), h1 + oH - 1)
  oW = w2 - w1
  oH = h2 - h1
  input = image.crop(input, w1, h1, w2, h2)
  assert(input:size(2) == oH)
  assert(input:size(3) == oW)

  -- update bboxes
  for b = 1,loc:size(1) do
    loc[{b,1}] = math.max(1, loc[{b,1}] - w1) -- x
    loc[{b,2}] = math.max(1, loc[{b,2}] - h1) -- y
  end

  -- Then scale to load size.
  local iW_orig = input:size(3)
  local iH_orig = input:size(2)
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

  out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]
  input_dbg = util.draw_keypoints(out:clone(), loc)

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
  local batch_info = {}
  while #batch_info < quantity do
    local ix_file = torch.randperm(#cur_files)[1]
    local info = torch.load(cur_files[ix_file])

    if opt.trainfiles ~= '' then
      if (info.has_kp[{1,1}] == 1) and
         (info.points[{3,{},{}}]:sum() > 0) and
         (#info.single_person:size() == 2) and
         (info.single_person:size(2) == 1) and
         (act2ix[info.act_name] ~= nil) then
        batch_info[#batch_info + 1] = info
      end
    else
      if (info.has_kp[{1,1}] == 1) and
         (info.points[{3,{},{}}]:sum() > 0) and
         (#info.single_person:size() == 2) and
         (info.single_person:size(2) == 1) and
         (cls2ix[info.cat_name] ~= nil) then
        batch_info[#batch_info + 1] = info
      end
    end
  end

  local data_img = torch.zeros(quantity, sampleSize[1], sampleSize[2], sampleSize[2])
  local data_dbg = torch.zeros(quantity, sampleSize[1], sampleSize[2], sampleSize[2])
  local data_txt = torch.zeros(quantity, opt.txtSize)
  local data_loc = torch.zeros(quantity, opt.num_elt, opt.keypoint_dim, opt.keypoint_dim)
  local loc_raw = torch.zeros(quantity, opt.num_elt, 3)

  for n = 1, quantity do
    local info = batch_info[n]
    local img_file = opt.img_dir .. '/' .. info.img

    -- randomly select a single person
    local person_ix = torch.randperm(info.single_person:size(2))[1]
    person_ix = info.single_person[{1,person_ix}]

    -- normalize image and keypoint locations.
    local img, locs, img_dbg = trainHook(img_file,
                                         info.points[{{},{},person_ix}],
                                         info.pos[{{},person_ix}],
                                         info.scale[{{},person_ix}],
                                         info.head[{{},person_ix}])
    data_img[{n,{},{},{}}]:copy(img)
    loc_raw[n]:copy(locs)

    -- add text information.
    local ix_txt = torch.randperm(info.txt:size(1) - opt.num_holdout)
    for c = 1,opt.numCaption do
      data_txt[n]:add(info.txt[ix_txt[c]]:double())
    end
    data_txt[n]:mul(1.0/opt.numCaption)

    for s = 1,opt.num_elt do
      local point = locs[{s,{}}]
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
  loc_raw:narrow(3,1,2):mul(opt.fineSize)
  return data_img, data_txt, data_loc, data_dbg, loc_raw
end

function trainLoader:size()
  return size
end

