
-- http://www.softcomplex.com/products/tigra_calendar/

local html = require 'orbiter.html'

-- if extensions use the bridge dispatch_static, then Orbit applications can
-- use this as well!
local bridge = require 'orbiter.bridge'
bridge.dispatch_static('/resources/javascript/calendar.+',
'/resources/css/calendar.+',
 '/resources/images/calendar.+')

local _M = {}
_M.mode = 'us'

html.set_defaults {
    scripts = '/resources/javascript/calendar.js',
    styles = '/resources/css/calendar.css'
}

function _M.set_mode(mode)
    _M.mode = mode
end

function _M.calendar(form,control,mode)
    return html.script ( ([[
 	new tcal ({
		formname: '%s',
		controlname: '%s',
        mode: '%s'
	});
    ]]):format(form,control,mode) )
end

local input = html.tags 'input'

function _M.date(form,control)
    return input{type='text',name=control},_M.calendar(form,control,_M.mode)
end

return _M
