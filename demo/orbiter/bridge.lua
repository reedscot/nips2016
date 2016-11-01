-- Orbiter, a personal web application framework
-- orbiter.orbit acts as a bridge for registering static dispatches
-- that works with both Orbiter and Orbit

local _M = {}

local app

function _M.new(app_)
    app = app_
end

function _M.dispatch_static(...)
    -- remember to strip off the starting @
    local path = debug.getinfo(2, "S").source:sub(2):gsub('\\','/')
    if path:find '/' then
        path = path:gsub('/[%w_]+%.lua$','')
    else -- invoked just as script name
        path = '.'
    end
    if orbit then
        local function static_handler(web)
            local fpath = path..web.path_info
            return app:serve_static(web,fpath)
        end
        app:dispatch_get(static_handler,...)
        return app
    else
        local obj = require 'orbiter'. new()
        obj.root = path
        obj:dispatch_static(...)
        return obj
    end
end

return _M


