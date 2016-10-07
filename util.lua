local util = {}


-- returns bbox in normalized coordinates.
function util.affine_to_bbox(imgdim, affine)
  local bbox = torch.zeros(4, affine:size(1))
  for b = 1,bbox:size(2) do
    local scale_Y = affine[{b,1,1}]
    local scale_X = affine[{b,2,2}]
    local t_y = affine[{b,1,3}]
    local t_x = affine[{b,2,3}]

    bbox[{1,b}] = 0.5 + t_x / 2.0 - scale_X / 2.0
    bbox[{2,b}] = 0.5 + t_y / 2.0 - scale_Y / 2.0
    bbox[{3,b}] = scale_X
    bbox[{4,b}] = scale_Y
  end
  return bbox
end

-- assumes bbox is in normalized coordinates.
function util.bbox_to_affine(imgdim, bbox)
  local affine = torch.zeros(bbox:size(2), 2, 3)
  for b = 1,bbox:size(2) do
    local b_x = bbox[{1,b}]
    local b_y = bbox[{2,b}]
    local bW =  loc[{3,b}]
    local bH =  loc[{4,b}]
    local t_x = 2*((b_x + bW/2.0) - 0.5)
    local t_y = 2*((b_y + bH/2.0) - 0.5)
    local scale_X = bW
    local scale_Y = bH
    loc_affine[{b,1,1}] = scale_Y
    loc_affine[{b,2,2}] = scale_X
    loc_affine[{b,1,3}] = t_y
    loc_affine[{b,2,3}] = t_x
  end
  return affine
end

function util.invert_affine(affine)
  local inv_aff = affine:clone()
  for b = 1,affine:size(1) do
    inv_aff[{b,1,1}] = 1.0/affine[{b,1,1}]
    inv_aff[{b,2,2}] = 1.0/affine[{b,2,2}]
    inv_aff[{b,1,3}] = -affine[{b,1,3}] / affine[{b,1,1}]
    inv_aff[{b,2,3}] = -affine[{b,2,3}] / affine[{b,2,2}]
  end
  return inv_aff
end

function util.normalize_box(img, box)
  local w_scale = box[3] / img:size(3)
  local h_scale = box[4] / img:size(2)
  box[1] = box[1] * w_sc
  box[2] = box[2] * h_scale
  box[3] = w_scale
  box[4] = h_scale
  return box
end

-- assumes bbox is in normalized coordinates.
function util.draw_all_boxes(img, bbox, thickness)
  return img
end

function util.draw_keypoints(img, points, sz, thickness)
  sz = sz or 0.06
  thickness = thickness  or 1
  local iW = img:size(3)
  local bW = math.floor(sz*iW*0.5)
  local iH = img:size(2)
  local bH = math.floor(sz*iH*0.5)

  for b = 1,points:size(1) do
    if points[{b,3}] > 0.5 then
      local box = torch.zeros(4)
      box[1] = (points[{b,1}] - bW) / iW
      box[2] = (points[{b,2}] - bH) / iH
      box[3] = bW*2/iW
      box[4] = bH*2/iH
      img = util.draw_box(img, box, thickness)
    end
  end
  return img
end

-- assumes bbox is in normalized coordinates.
function util.draw_box(img, bbox, thickness)
  thickness = thickness or 3
  local boxcolor = torch.zeros(3)
  boxcolor[3] = 1

  local iW = img:size(3)
  local iH = img:size(2)

  local x = math.floor(bbox[1] * iW)
  local y = math.floor(bbox[2] * iH)
  local width = math.floor(bbox[3] * iW)
  local height = math.floor(bbox[4] * iH)

  -- top bar
  for n = 1,width do
    for t = 1,thickness do
      local offset = -math.floor(thickness / 2) + (t-1)
      if ((y + offset) <= img:size(2) and  (y + offset) >= 1) then
        if (x+n-1 <= img:size(3) and (x+n-1 > 0)) then
          img[{{},y + offset, x+n-1}]:copy(boxcolor)
        end
      end
    end
  end
  -- bottom bar
  for n = 1,width do
    for t = 1,thickness do
      local offset = -math.floor(thickness / 2) + (t-1)
      if ((y + height + offset) <= img:size(2) and (y + height + offset) >= 1) then
        if (x+n-1 <= img:size(3) and (x+n-1 > 0)) then
          img[{{},y + height + offset, x+n-1}]:copy(boxcolor)
        end
      end
    end
  end
  -- left bar
  for n = 1,height do
    for t = 1,thickness do
      local offset = -math.floor(thickness / 2) + (t-1)
      if ((x + offset) <= img:size(3) and (x + offset) >= 1) then
        if (y+n-1 <= img:size(2) and (y+n-1 > 0)) then
          img[{{},y+n-1, x + offset}]:copy(boxcolor)
        end
      end
    end
  end
  -- right bar
  for n = 1,height do
    for t = 1,thickness do
      local offset = -math.floor(thickness / 2) + (t-1)
      if ((x + width + offset) <= img:size(3) and (x + width + offset) >= 1) then
        if (y+n-1 <= img:size(2) and (y+n-1 > 0)) then
          img[{{},y+n-1, x + width + offset}]:copy(boxcolor)
        end
      end
    end
  end

  return img
end

return util

