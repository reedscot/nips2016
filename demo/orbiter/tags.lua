----
-- @module tags
local html = require 'orbiter.html'

local _M = {}

local html5_tags = {
   a = true,
   abbr = true,
   address = true,
   area = true,
   article = true,
   aside = true,
   audio = true,
   b = true,
   base = true,
   bdi = true,
   bdo = true,
   blockquote = true,
   body = true,
   br = true,
   button = true,
   canvas = true,
   caption = true,
   cite = true,
   code = true,
   col = true,
   colgroup = true,
   command = true,
   data = true,
   datagrid = true,
   datalist = true,
   dd = true,
   del = true,
   details = true,
   dfn = true,
   div = true,
   dl = true,
   dt = true,
   em = true,
   embed = true,
   eventsource = true,
   fieldset = true,
   figcaption = true,
   figure = true,
   footer = true,
   form = true,
   h1 = true,
   h2 = true,
   h3 = true,
   h4 = true,
   h5 = true,
   h6 = true,
   head = true,
   header = true,
   hgroup = true,
   hr = true,
   html = true,
   i = true,
   iframe = true,
   img = true,
   input = true,
   ins = true,
   kbd = true,
   keygen = true,
   label = true,
   legend = true,
   li = true,
   link = true,
   mark = true,
   map = true,
   menu = true,
   meta = true,
   meter = true,
   nav = true,
   noscript = true,
   object = true,
   ol = true,
   optgroup = true,
   option = true,
   output = true,
   p = true,
   param = true,
   pre = true,
   progress = true,
   q = true,
   ruby = true,
   rp = true,
   rt = true,
   s = true,
   samp = true,
   script = true,
   section = true,
   select = true,
   small = true,
   source = true,
   span = true,
   strong = true,
   style = true,
   sub = true,
   summary = true,
   details = true,
   sup = true,
   table = true,
   tbody = true,
   td = true,
   textarea = true,
   tfoot = true,
   th = true,
   thead = true,
   time = true,
   title = true,
   tr = true,
   track = true,
   u = true,
   ul = true,
   var = true,
   video = true,
   wbr = true,
}

local tag,rawget = html.tags,rawget
local env,_G = {html=html},_G

_M.env = {
    __index = function(t,name)
        local g = rawget(_G,name) -- in case of strict mode!
        if g then return g end
        -- see if it's an HTML tag!
        name = name:lower()
        if html5_tags[name] then
            env[name] = tag(name)
        else
            error("unknown HTML tag "..name,2)
        end
        return rawget(env,name)
    end
}

setmetatable(env,_M.env)

function _M.use ()
    setfenv(2,env)
end

function _M.set (funs)
    for _,f in ipairs(funs) do
        setfenv(f,env)
    end
end

function _M.register (obj)
    obj.env = env
end

return _M
