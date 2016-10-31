local orbiter = require 'orbiter'
local html = require 'orbiter.html'
local text = require 'orbiter.text'
local bridge = require 'orbiter.bridge'
local _M = {}

local jquery_js = '/resources/jquery-1.8.3.min.js'

local app = bridge.dispatch_static(text.lua_escape(jquery_js))

local ajax_request = '/jq/request'

html.set_defaults {
   scripts = jquery_js,
   inline_script = ([[
    function jq_call_server(id,klass,tid) {
        $.get("%s",{id: id, group_id: tid, klass: klass });
    }
    function jq_file_click(klass,id,tid,event) {
        var href = $('#'+id).attr('title');
        if (href == "") {
            href = $('#'+id+' span').attr('title');
        }
        if (href != "" && href != undefined) {
            window.location = href;
        } else { // otherwise, it's for the server to handle
            jq_call_server(id,klass,tid)
        }
        event.stopImmediatePropagation();
        return false;
    }
    function jq_set_click(select,container_id) {
        $(select).click(function(event) {
            var klass = $(this).attr('class')
            return jq_file_click(klass,this.id,container_id,event);
        })
    }
    function jq_submit_form(id) {
        var form = $("form#"+id);
        $.post(form.attr("action"), form.serialize());
    }
]]):format(orbiter.prepend_root(ajax_request)),
}

---- some useful functions ----

-- minimal escaping for JS strings
function _M.escape(s)
    return tostring(s):gsub("'","\\'"):gsub("\n","\\n")
end

function _M.eval (s)
    return s, "text/javascript"
end

function _M.alert(s)
    return _M.eval('alert("'.._M.escape(s)..'");')
end

local lua_data_map = {}
local lua_data_index = 1

function _M.data_to_id(data)
    local ret = ('D%x'):format(lua_data_index)
    lua_data_index = lua_data_index + 1
    lua_data_map[ret] = data
    return ret
end
local data_to_id = _M.data_to_id

local function call_if(fun,arg,id)
    if fun then
        local resp,mtype = fun(arg,id)
        mtype = mtype or 'text/javascript'
        return tostring(resp),mtype
    else
        return ''
    end
end

function _M.call_handler(idata,tdata,id,name)
    return call_if(idata[name] or tdata[name],idata,id)
end

-- callback magic: if an item is clicked, then the JS function jq_file_click
-- will make an async request which is handled here.
-- An application may add to this functionality....
-- (By default, callbacks are assumed to return JavaScript; set another mime type
-- explicitly if this is inappropriate.)
function app:jq_request(web)
    local vars = web.input
    if orbiter.tracing then print('request wuz ',vars.id, vars.group_id,vars.klass) end
    local klass = vars.klass or 'nada'
    local idata = lua_data_map[vars.id] or 'nada'
    local tdata = lua_data_map[vars.group_id] or 'nada'
    if idata == 'nada' and tdata == 'nada' then
        print('received nada',vars.klass,vars.id,vars.group_id)
        return ''
    end
    local resp,mtype
    if tdata.click_handler then
        local resp,mtype = tdata.click_handler(klass,idata,tdata,vars.id)
        if resp then return resp,mtype end
    end
    return _M.call_handler(idata,tdata,vars.id,'click')
end

app:dispatch_get(app.jq_request,ajax_request)

function _M.set_data(id,data)
    local this = { data = data }
    lua_data_map[id] = this
    return this
end

function _M.set_handler (name, id, obj)
    if obj[name] then
        lua_data_map[id][name] = obj[name]
        obj[name] = nil
    end
end

local button_  = html.tags 'button'

local function as_callback (callback)
    if type(callback) == 'table' then
        local id = callback.id
        callback = function()
            return 'jq_submit_form("'..id..'");'
        end
    end
    return callback
end

function _M.button(label,callback)
    return {
        button_{class='click-button',
            id = data_to_id {click = as_callback(callback)},
            label},
        html.script('jq_set_click("button.click-button","buttons")')
    }
end

function _M.link(label,callback)
    return {
        span{class='click-link',
            id = data_to_id { click = callback },
            label},
        html.script('jq_set_click("span.click-link","buttons")')
    }
end

function _M.reload()
    return "window.location.href=window.location.href"
end

local JMT = {}

local function has_meta(t,name)
    local mt = getmetatable(t)
    if not mt then return false
    else
        return mt[name] ~= nil
    end
end

local js_tostring

function js_tostring(args)
    local concat = table.concat
    for i,a in ipairs(args) do
        local ta = type(a)
        if html.is_doc(a) then
            -- don't want pretty-printing here!
            a = html.raw_tostring(a)
            ta = 'string'
        end
        if ta == 'table' and not has_meta(a,'__tostring') then
            if #a > 0 then -- list-like
                return '['..js_tostring(a)..']'
            else -- map-like
                local vv = {}
                for k,v in pairs(a) do
                    vv[#vv+1] = k..' : '..js_tostring {v}
                end
                a = '{'..concat(vv,',')..'}'
            end
        elseif ta == 'string' then
            a = '"'.._M.escape(a)..'"'
        else
            a = tostring(a)
        end
        args[i] = a
    end
    return concat(args,',')
end

function _M.tostring(t)
    return js_tostring{t}
end

local function jqw(code)
    return setmetatable({js = code},JMT)
end

-- this module is callable, e.g. jq("#iden") and creates a jQuery selector
-- expression
setmetatable(_M,{
    __call = function(tbl,selector)
        return jqw ('$("'..selector..'")' )
    end
})

function JMT.__tostring(self)
    return self.js
end

function JMT.__index(self,method)
    if method == '_' then method = ';'
    elseif method == '_end' then method = 'end'
    end
    return jqw(self.js..'.'..method)
end

function JMT.__call(obj,self,...)
    local args = {...} -- assume no nils
    return jqw(obj.js..'('..js_tostring(args,',')..')')
end

function _M.timeout(ms,callback)
    local id = data_to_id { click = as_callback(callback) }
    return ([[
        setTimeout("jq_call_server('%s',null,null)", %d)
    ]]):format(id,ms)
end

function _M.timeout_script (ms,callback)
    return html.script(_M.timeout(ms,callback))
end

function _M.use_timer()
    html.set_defaults {
        inline_script = [[
        var jq_timer_data = [null,null];
        function jq_timer() {
            if (jq_timer_data[0] != null) {
                jq_call_server(jq_timer_data[0],null,null);
                setTimeout("jq_timer()", jq_timer_data[1]);
            }
        }
    ]]
    }
end

function _M.timer(ms,callback)
    local id = data_to_id { click = as_callback(callback) }
    return ('jq_timer_data = ["%s",%d];setTimeout("jq_timer()", %d)'):format(id,ms,ms)
end

function _M.cancel_timer()
    return 'jq_timer_data[0] = null'
end


return _M
