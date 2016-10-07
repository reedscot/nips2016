
local CropInvert, parent = torch.class('nn.CropInvert', 'nn.Module')

function CropInvert:__init()
   parent.__init(self)
end

function CropInvert:updateOutput(input)
  self.output:resizeAs(input):fill(0)
  for b = 1,input:size(1) do
    self.output[{b,1,1}] = 1.0/input[{b,1,1}]
    self.output[{b,2,2}] = 1.0/input[{b,2,2}]
    self.output[{b,1,3}] = -input[{b,1,3}] / input[{b,1,1}]
    self.output[{b,2,3}] = -input[{b,2,3}] / input[{b,2,2}]
  end
  return self.output
end

function CropInvert:updateGradInput(input, gradOutput)
  self.gradInput = input:clone():fill(0)
  for b = 1,input:size(1) do
    local s_x = input[{b,1,1}]
    local d_s_x = gradOutput[{b,1,1}]
    local s_y = input[{b,2,2}]
    local d_s_y = gradOutput[{b,2,2}]
    local t_x = input[{b,1,3}]
    local d_t_x = gradOutput[{b,1,3}]
    local t_y = input[{b,2,3}]
    local d_t_y = gradOutput[{b,2,3}]

    self.gradInput[{b,1,1}] = -d_s_x/math.pow(s_x,2) + d_t_x*t_x/math.pow(s_x,2)
    self.gradInput[{b,2,2}] = -d_s_y/math.pow(s_y,2) + d_t_y*t_y/math.pow(s_y,2)
    self.gradInput[{b,1,3}] = d_t_x * (-1/s_x)
    self.gradInput[{b,2,3}] = d_t_y * (-1/s_y)
  end
  return self.gradInput
end

