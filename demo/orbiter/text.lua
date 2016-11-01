-- Orbiter, a personal web application framework
-- Yet another little template expansion function, but dead simple (promise!).
-- Also supports Python-like string formatting by overloading % operator.
-- (see http://lua-users.org/wiki/StringInterpolation
local _M  = {}

function _M.lua_escape(s)
    return (s:gsub('[%-%.%+%[%]%(%)%$%^%%%?%*]','%%%1'))
end

local function basic_subst(s,t)
    return (s:gsub('%$([%w_]+)',t))
end

local format = string.format

--- version of `string.format` where '%s' uses `tostring`
function _M.formatx (fmt,...)
    local args = {...}
    local i = 1
    for p in fmt:gmatch('%%.') do
        if p == '%s' and type(args[i]) ~= 'string' then
            args[i] = tostring(args[i])
        end
        i = i + 1
    end
    return format(fmt,unpack(args))
end

--- Python-like string formatting with % operator.
-- Note this goes further than the original, and will allow these cases:
-- 1. a single value
-- 2. a list of values
-- 3. a map of var=value pairs
-- 4. a function, as in gsub
-- For the second two cases, it uses $-variable substituion.
function _M.enable_python_formatting()
    local formatx = _M.formatx
    getmetatable("").__mod = function(a, b)
        if b == nil then
            return a
        elseif type(b) == "table" and getmetatable(b) == nil then
            if #b == 0 then -- assume a map-like table
                return basic_subst(a,b)
            else
                return formatx(a,unpack(b))
            end
        elseif type(b) == 'function' then
            return basic_subst(a,b)
        else
            return formatx(a,b)
        end
    end
end

--- really basic templates;
-- (Templates are callable so subst is unnecessary)
--
--     t = text.Template 'hello $world'
--     print(t:subst {world = 'dolly'}).
function _M.Template(str)
    local tpl = {s=str}
    function tpl:subst(t)
        return basic_subst(str,t)
    end
    setmetatable(tpl,{
        __call = function(obj,t)
            return obj:subst(t)
        end
    })
    return tpl
end

_M.subst = basic_subst

return _M -- orbiter.text
