
-- http://www.flotcharts.org

local html = require 'orbiter.html'
local jq = require 'orbiter.libs.jquery'

require 'orbiter.bridge'.dispatch_static('/resources/javascript/jquery.+')

local flot = {}

html.set_defaults {
    scripts = '/resources/javascript/jquery.flot.min.js',
}

local interactive

local function set_interactive ()
    if not interactive then
        html.set_defaults {
            scripts = '/resources/javascript/jquery.flot.navigate.min.js',
        }
        interactive = true
    end
end

local function interleave (xv,yv)
    local res = {}
    for i = 1,#xv do
        res[i] = {xv[i],yv[i]}
    end
    return res
end

function flot.range (x1,x2,incr)
    local res, i = {}, 1
    for x = x1,x2,incr do
        res[i] = x
        i = i + 1
    end
    return res
end

local concat,append = table.concat,table.insert
local as_js

flot.null = setmetatable({},{
   __tostring = function(self) return "null" end
})

-- you can of course use any available Lua JSON library here - this is good
-- enough for our purposes.
function as_js (t)
   local mt = getmetatable(t)
   if type(t) ~= 'table' or (mt and mt.__tostring) then
      return type(t) == 'string' and '"'..t..'"' or tostring(t)
   elseif #t > 0 then -- it's an array!
      local res = {}
      for i = 1,#t do
         res[i] = as_js(t[i])
      end
      return '['..concat(res,',')..']'
   else
      local res = {}
      for k,v in pairs(t) do
         append(res,k..':'..as_js(v))
      end
      return '{'..concat(res,',')..'}'
   end
end

local kount = 0
local div = html.tags 'div'

local script = [[
var plotvar_%s
function plot_%s (data) {
    plotvar_%s = $.plot($("#%s"),data,%s);
}
$(function () {
    plot_%s (%s);
});
]]

function flot.Plot (opts)
    local plot = {}
    kount = kount + 1
    plot.idx = "flotdiv"..kount
    opts = opts or {}
    plot.width = opts.width or 600
    plot.height = opts.height or 400
    plot.xvalues = opts.xvalues
    opts.width = nil -- no harm, but they're not valid options.
    opts.height = nil
    opts.xvalues = nil

    -- navigation plugin
    if opts.interactive ~= nil then
        opts.zoom = {interactive=opts.interactive}
        opts.pan = {interactive=opts.interactive}
        opts.interactive = nil
    end
    if opts.zoom or opts.pan then
        set_interactive ()
    end

    local dataset = {}

    function plot:show ()
        local id, data, options = self.idx, as_js(dataset), as_js(opts)
        --local code = render_script(self.idx,as_js(dataset),as_js(opts))
        return div {
            div {id=self.idx, style=('width:%spx;height:%spx'):format(self.width,self.height),''},
            html.script(script:format(id,id,id,id,options,id,data))
        }
    end

    function plot:update ()
        return ('plot_%s(%s);'):format(self.idx, as_js(dataset))
    end

    function plot:clear ()
        dataset = {}
    end

    local Series = {}
    Series.__index = Series

    function Series:update(data)
        if data.x then
            data = interleave(data.x,data.y)
        elseif plot.xvalues and type(data[1]) ~= 'table' then
            data = interleave(plot.xvalues,data)
        end
        self.data = data
    end

    local function new_series (label,data,kind)
        local series = setmetatable(kind or { lines = { show = true }},Series)
        series.label = label
        if data then
            series:update(data)
        end
        return series
    end

    function plot:add_series(label,data,kind)
        local series = new_series(label,data,kind)
        append(dataset,series)
        return series
    end

    local function plotmethod (name)
        plot[name] = function(self)
            return 'plotvar_'..self.idx..'.'..name..'()'
        end
    end

    if interactive then
        plotmethod 'zoomOut'
        plotmethod 'zoom'
        plotmethod 'pan'
    end


    return plot
end


return flot

