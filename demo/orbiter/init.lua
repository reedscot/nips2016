--- Orbiter, a compact personal web application framework.
-- Very much inspired by Orbit, with a little webserver borrowed from Webrocks

local socket = require 'socket.core'

local _M = {}  --- our module
local DIRSEP = package.config:sub(1,1)
local Windows = DIRSEP == '\\'
local t_remove, t_insert, append = table.remove, table.insert, table.insert
local tracing, browser
local help_text = [[
 **** Orbiter v0.3 ****
--addr=IP address (default localhost)
--port=HTTP port (default 8080)
--trace   print out some useful verbosity
--test=pattern  print out what will be sent to the user agent
--no_headers when using --test, don't print out HTTP headers
--launch  open the browser; may point to a particular browser, otherwise system.
]]

local function quit(msg)
    io.stderr:write(msg,'\n')
    os.exit(1)
end

-- sort out Lua 5.2 compatibility in a brutal way..
if not unpack then
    rawset(_G,'unpack',rawget(table,'unpack'))
end

local trace

--- Extract flags from an arguments list.
-- (grabbed from luarocks.util)
-- Given string arguments, extract flag arguments into a flags set.
-- For example, given "foo", "--tux=beep", "--bla", "bar", "--baz",
-- it would return the following:
-- {["bla"] = true, ["tux"] = "beep", ["baz"] = true}, "foo", "bar".
function _M.parse_flags(...)
   local args = {...}
   local flags = {}
   for i = #args, 1, -1 do
      local flag = args[i]:match("^%-%-(.*)")
      if flag then
         local var,val = flag:match("([a-z_%-]*)=(.*)")
         if val then
            flags[var] = val
         else
            flags[flag] = true
         end
         t_remove(args, i)
      end
   end
   return flags, unpack(args)
end

local function readfile (f)
	local f,err = io.open(f, 'rb')
	if not f then return nil,err end
	local s = f:read '*a'
	f:close()
	return s
end

local MT = {}
MT.__index = MT

-- this is the object used by Orbiter itself to provide one basic piece of furniture,
-- the favicon.
local self

function _M.new(...)
    local extensions = {...}
    local obj = extensions[1]
    -- if passed a table which doesn't have register, then assume we're being called
    --    from module()
    -- use the module as the object, and manually add our methods to it.
    if obj and type(obj) ~= 'string' and not obj.register then
        local m = extensions[1]
        for k,v in pairs(MT) do m[k] = v end
        table.remove(extensions,1)
    else
        obj = setmetatable({},MT)
    end
    if not self then
        self = true -- hack to prevent endless recursion
        self = _M.new()
        self:dispatch_static '/resources/favicon%.ico'
    end
    -- remember to strip off the starting @
    local path = debug.getinfo(2, "S").source:sub(2):gsub('\\','/')
    if path:find '/' then
        path = path:gsub('/[%w_]+%.lua$','')
    else -- invoked just as script name
        path = '.'
    end
    obj.root = path
    obj.resources = obj.root..'/resources'
    if #extensions > 0 then
        for _,e in ipairs(extensions) do
            if type(e) == 'string' then
                local stat,ext = pcall(require,'orbiter.'..e)
                if not stat then quit("cannot load extension: "..e) end
                e = ext
            end
            e.register(obj)
        end
    end
    return obj
end



---- launch the browser ----

local browsers = {
    "x-www-browser","gnome-open","xdg-open"
}

local function shell(cmd)
    local f = io.popen 'uname -s'
    local line = f:read()
    f:close()
    return line
end

local function uname()
    return shell 'uname s'
end

local function which(prog)
    return shell ('which %s 2> /dev/null' % prog)
end

local function launch_browser (url,browser)
    if browser == true then
        browser = nil -- autodetect!
    end
    if Windows then
        os.execute('rundll32 url.dll,FileProtocolHandler '..url)
        return
    end
    if not browser then
        local line = uname()
        if line == 'Darwin' then
            browser = 'open'
        else
            for _,p in ipairs(browsers) do
                if which(p) then browser = p; break end
            end
        end
    end
    os.execute(browser..' '..url..'&')
end

----- Root namespace support ---------

local function starts_with (path,prefix)
    local i1,i2 = path:find(prefix,1,true)  -- plain match
    return i1 ~= nil and i1 == 1 and i2 == #prefix
end

function _M.set_root (path)
    _M.ROOT = '/'..path
    _M.SUBSTITUTIONS = {ROOT=_M.ROOT}
end

setmetatable(_M,{
    __call = function(self,path)
        _M.set_root(path)
        return _M
    end
})

function _M.prepend_root (path)
    if _M.ROOT and not (path=='#' or path:match '^http[s]://' or starts_with(path,_M.ROOT)) then
        path = _M.ROOT..path
    end
    return path
end

-- used to get actual path for static file lookup
function _M.strip_root (file)
    if _M.ROOT then
        if file == _M.ROOT then
            file = '/'
        else
            local k
            file,k = file:gsub(_M.ROOT,'',1)
            if k == 0 then error("request was NOT namespaced! "..file) end
        end
    end
    return file
end

-- hack to let /root as well as /root/ be recognized
local function check_root (file)
    if _M.ROOT and file == _M.ROOT then
        file = file .. '/'
    end
    return file
end

----- URL pattern dispatch --------------

local dispatch_set_handler

local function static_handler(obj,web)
    local content,mime = obj:read_content(web.path_info)
    if not content then
        return '404 Not Found',false
    else
        return content,mime,{
            ["Cache-Control"] = "max-age=86400"
        }
    end
end

function MT:dispatch_get(callback,...)
    dispatch_set_handler('GET',self,callback,...)
end

function MT:dispatch_post(callback,...)
    dispatch_set_handler('POST',self,callback,...)
end

function MT:dispatch_any(callback,...)
    dispatch_set_handler('*',self,callback,...)
end

function MT:dispatch_static(...)
    dispatch_set_handler('GET',self,static_handler,...)
end

local patterns = {}

function dispatch_set_handler(method,obj,callback,...)
    local pats = {...}
    if #pats == 1 then
        local pat = pats[1]
        assert(type(pat) == 'string')
        pat = _M.prepend_root(pat)
        if tracing then print('pattern '..pat) end
        pat = '^'..pat..'$'
        local pat_rec = {pat=pat,callback=callback,self=obj,method=method}
        -- objects can override existing patterns, so we look for this pattern
        -- and replace the handler, if the method is the same
        local idx
        for i,p in ipairs(patterns) do
            if p.pat == pat and p.method == method then idx = i; break  end
        end
        if idx then
            patterns[idx] = pat_rec
        else
            append(patterns,pat_rec)
        end
    else
        for _,pat in ipairs (pats) do
            dispatch_set_handler(method,obj,callback,pat)
        end
    end
end

-- The idea here is that if multiple patterns match an url,
-- then pick the longest such pattern.
-- Very general patterns (like /(.-)(/.*)) can be long, but very general.
-- So we use the length after stripping out any magic characters.
-- returns the callback, the pattern captures, and the object (if any)
local function match_patterns(method,request,obj)
    local max_pat, max_i = 0
    local max_captures
    request = check_root(request)
    for i = 1,#patterns do
        local tpat = patterns[i]
        if (tpat.method == '*' or tpat.method == method) and (obj==nil or tpat.self==obj) then
            local pat = tpat.pat
            if tracing == 'all' then print('trying',pat,request) end
            local captures = {request:match(pat)}
            local pat_size = #(pat:gsub('[%(%)%.%+%-%*]',''))
            if #captures > 0 and pat_size > max_pat then
                max_i = i
                max_pat = pat_size
                max_captures = captures
                if tracing then trace('matching '..pat..' '..pat_size) end
            end
        end
    end
    if max_captures then
        return patterns[max_i].callback,max_captures,patterns[max_i].self
    end
end

local request_filters = {}

local function process_request_filters(web,file)
    for _,f in ipairs(request_filters) do
        local newp,obj = f(web,file)
        if newp then return newp,obj end
    end
    return file
end

function _M.add_request_filter(f)
    append(request_filters,f)
end

function _M.remove_request_filter(f)
    local idx
    for i, ff in ipairs(request_filters) do
        if ff == f then idx = i; break end
    end
    if idx then table.remove(request_filters,idx) end
end

function _M.get_pattern_table()
    return patterns
end


--------------- HTTP Server --------------------
----  A little web server, based on code by Samuel Saint-Pettersen ----
-- headers and error handling much improved by Ignacio

local mime_types = {
    other = 'text/plain',
	ez = "application/andrew-inset",
	atom = "application/atom+xml",
	hqx = "application/mac-binhex40",
	cpt = "application/mac-compactpro",
	mathml = "application/mathml+xml",
	doc = "application/msword",
	bin = "application/octet-stream",
	dms = "application/octet-stream",
	lha = "application/octet-stream",
	lzh = "application/octet-stream",
	exe = "application/octet-stream",
	class = "application/octet-stream",
	so = "application/octet-stream",
	dll = "application/octet-stream",
	dmg = "application/octet-stream",
	oda = "application/oda",
	ogg = "application/ogg",
	pdf = "application/pdf",
	ai = "application/postscript",
	eps = "application/postscript",
	ps = "application/postscript",
	rdf = "application/rdf+xml",
	smi = "application/smil",
	smil = "application/smil",
	gram = "application/srgs",
	grxml = "application/srgs+xml",
	mif = "application/vnd.mif",
	xul = "application/vnd.mozilla.xul+xml",
	xls = "application/vnd.ms-excel",
	ppt = "application/vnd.ms-powerpoint",
	rm = "application/vnd.rn-realmedia",
	wbxml = "application/vnd.wap.wbxml",
	wmlc = "application/vnd.wap.wmlc",
	wmlsc = "application/vnd.wap.wmlscriptc",
	vxml = "application/voicexml+xml",
	bcpio = "application/x-bcpio",
	vcd = "application/x-cdlink",
	pgn = "application/x-chess-pgn",
	cpio = "application/x-cpio",
	csh = "application/x-csh",
	dcr = "application/x-director",
	dir = "application/x-director",
	dxr = "application/x-director",
	dvi = "application/x-dvi",
	spl = "application/x-futuresplash",
	gtar = "application/x-gtar",
	hdf = "application/x-hdf",
	xhtml = "application/xhtml+xml",
	xht = "application/xhtml+xml",
	js = "application/x-javascript",
	skp = "application/x-koan",
	skd = "application/x-koan",
	skt = "application/x-koan",
	skm = "application/x-koan",
	latex = "application/x-latex",
	xml = "application/xml",
	xsl = "application/xml",
	dtd = "application/xml-dtd",
	nc = "application/x-netcdf",
	cdf = "application/x-netcdf",
	sh = "application/x-sh",
	shar = "application/x-shar",
	swf = "application/x-shockwave-flash",
	xslt = "application/xslt+xml",
	sit = "application/x-stuffit",
	sv4cpio = "application/x-sv4cpio",
	sv4crc = "application/x-sv4crc",
	tar = "application/x-tar",
	tcl = "application/x-tcl",
	tex = "application/x-tex",
	texinfo = "application/x-texinfo",
	texi = "application/x-texinfo",
	t = "application/x-troff",
	tr = "application/x-troff",
	roff = "application/x-troff",
	man = "application/x-troff-man",
	me = "application/x-troff-me",
	ms = "application/x-troff-ms",
	ustar = "application/x-ustar",
	src = "application/x-wais-source",
	zip = "application/zip",
	au = "audio/basic",
	snd = "audio/basic",
	mid = "audio/midi",
	midi = "audio/midi",
	kar = "audio/midi",
	mpga = "audio/mpeg",
	mp2 = "audio/mpeg",
	mp3 = "audio/mpeg",
	aif = "audio/x-aiff",
	aiff = "audio/x-aiff",
	aifc = "audio/x-aiff",
	m3u = "audio/x-mpegurl",
	ram = "audio/x-pn-realaudio",
	ra = "audio/x-pn-realaudio",
	wav = "audio/x-wav",
	pdb = "chemical/x-pdb",
	xyz = "chemical/x-xyz",
	bmp = "image/bmp",
	cgm = "image/cgm",
	gif = "image/gif",
	ief = "image/ief",
	jpeg = "image/jpeg",
	jpg = "image/jpeg",
	jpe = "image/jpeg",
	png = "image/png",
	svg = "image/svg+xml",
	svgz = "image/svg+xml",
	tiff = "image/tiff",
	tif = "image/tiff",
	djvu = "image/vnd.djvu",
	djv = "image/vnd.djvu",
	wbmp = "image/vnd.wap.wbmp",
	ras = "image/x-cmu-raster",
	ico = "image/x-icon",
	pnm = "image/x-portable-anymap",
	pbm = "image/x-portable-bitmap",
	pgm = "image/x-portable-graymap",
	ppm = "image/x-portable-pixmap",
	rgb = "image/x-rgb",
	xbm = "image/x-xbitmap",
	xpm = "image/x-xpixmap",
	xwd = "image/x-xwindowdump",
	igs = "model/iges",
	iges = "model/iges",
	msh = "model/mesh",
	mesh = "model/mesh",
	silo = "model/mesh",
	wrl = "model/vrml",
	vrml = "model/vrml",
	ics = "text/calendar",
	ifb = "text/calendar",
	css = "text/css",
	html = "text/html",
	htm = "text/html",
	asc = "text/plain",
	txt = "text/plain",
	rtx = "text/richtext",
	rtf = "text/rtf",
	sgml = "text/sgml",
	sgm = "text/sgml",
	tsv = "text/tab-separated-values",
	wml = "text/vnd.wap.wml",
	wmls = "text/vnd.wap.wmlscript",
	etx = "text/x-setext",
	mpeg = "video/mpeg",
	mpg = "video/mpeg",
	mpe = "video/mpeg",
	qt = "video/quicktime",
	mov = "video/quicktime",
	mxu = "video/vnd.mpegurl",
	avi = "video/x-msvideo",
	movie = "video/x-sgi-movie",
	ice = "x-conference/x-cooltalk",
	rss = "application/rss+xml",
	atom = "application/atom+xml"
}

local function url_decode(url)
    url = url:gsub('%+',' ')
    url = url:gsub('%%(%x%x)',function(c)
        c = tonumber(c,16)
        return string.char(c)  -- ('%s'):format(?
    end)
    return url
end

local function url_encode(str)
  if not str then return nil end
  str = string.gsub (str, "\n", "\r\n")
  str = string.gsub (str, "([^%w ])",
        function (c) return string.format ("%%%02X", string.byte(c)) end)
  str = string.gsub (str, " ", "+")
  return str
end

local function url_split(vars)
    local res = {}
    for pair in vars:gmatch('[^&]+') do
        local k,v = pair:match('([^=]+)=(.*)')
        v = url_decode(v)
        if res[k] then -- multiple values for this name
            if type(res[k]) == 'string' then res[k] = {res[k]} end
            append(res[k],v)
        else
            res[k] = v
        end
    end
    return res
end

local function send_error(client, code, message)
	local header = "HTTP/1.1 " .. code .. "\r\nContent-Type:text/html\r\n"
	local msg = (
    [[<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">
<html><head>
<title>%s</title>
</head><body>
<h1>%s</h1>
<p>%s</p>
<hr/>
<small>Orbiter Web Server v1.0</small>
</body></html>]]):format(code, code, message or code)
	header = header .. "Content-Length:" .. #msg .. "\r\n\r\n"
	client:send(header)
	client:send(msg)
end

local function send_headers (client, code, type, length, sheaders)
    local extra = ''
    if sheaders then
        local res, append = {}, table.insert
        for head, value in pairs(sheaders) do
            append(res,head..": "..value..'\r\n')
        end
        extra = table.concat(res)
    end
	client:send( ("HTTP/1.1 %s\r\n%sContent-Type: %s\r\nContent-Length: %d\r\nConnection: close\r\n\r\n"):format(code, extra, type, length) )
end

-- process headers from a connection (borrowed from socket.http)
-- note that the variables have '-' replaced as '_' to make them WSAPI compatible
-- (e.g. HTTP_CONTENT_LENGTH)
local function receiveheaders(sock)
    local line, name, value, err
    local headers = {}
    -- get first line
    line, err = sock:receive()
    if err then return nil, err end
    -- headers go until a blank line is found
    while line ~= "" do
        -- get field-name and value
        name, value = line:match "^(.-):%s*(.*)"
        if not (name and value) then return nil, "malformed reponse headers" end
        name = 'HTTP_'..name:upper():gsub('%-','_')
        -- get next line (value might be folded)
        line, err  = sock:receive()
        if err then return nil, err end
        -- unfold any folded values
        while line:find("^%s") do
            value = value .. line
            line = sock:receive()
            if err then return nil, err end
        end
        -- save pair in table
        if headers[name] then headers[name] = headers[name] .. ", " .. value
        else headers[name] = value end
    end
    return headers
end

function MT:get_path_to(file)
    return self.root .. file
end

function MT:read_content(file)
    file = _M.strip_root(file)
    if file == '/' then file = '/index.html' end
    local extension = file:match('%.(%a+)$') or 'other'
    local path = self:get_path_to(file)
    local content = readfile (path)
    if content then
        if tracing then trace('returning '..path) end
        local mime = mime_types[extension] or 'text/plain'
        return content,mime
    else
        print('error: unable to find '..path)
        return false
    end
end

function MT:dispatch(web,path)
    local action,captures,obj = match_patterns('GET',path) --??
    if not action or obj ~= self then return nil end
    return action(obj,web,unpack(captures))
end

local running,last_obj

function _M.get_last_object()
    return last_obj
end

function trace(stuff)
    io.stderr:write(stuff,'\n')
end

local num_retries = 100

-- this is based on socket.bind from LuaSocket.
-- However, we _do_ want to know if the port is already in use,
-- because we want to try again with a higher port number.
local function socket_bind(host,port,backlog)
    local sock, err, res
    for i = 1,num_retries do
        sock, err = socket.tcp()
        if not sock then return nil, err end
        sock:setoption("reuseaddr", true)
        res, err = sock:bind(host, port)
        if not res then
            if err == 'address already in use' then
                port = port + 1
            else
                return nil,err
            end
        else
            break
        end
    end
    res, err = sock:listen(backlog)
    if not res then return nil, err end
    return sock,port
end

----- Web class -----
local Web =
{
	vars		= {},
	input		= {},
	headers		= {},
	status		= "200 OK",
	method		= "",
	path_info	= ""
}

function Web:redirect(url)
	self.status = "302 Found"
	self.headers["Location"] = _M.prepend_root(url)
	return "redirect"
end

function Web:link(url, params)
	local link = {}
	-- url encode parameters
	for k, v in pairs(params or {}) do
		link[#link + 1] = k .. "=" .. url_encode(v)
	end
	local qs = table.concat(link, "&")
	if qs and qs ~= "" then
		return _M.prepend_root(url) .. "?" .. qs
	else
		return _M.prepend_root(url)
	end
end

function Web:new(o)
	o = o or {}   -- create object if user does not provide one
	setmetatable(o, self)
	self.__index = self
	return o
end

function MT:run(...)
    local args,flags
    flags,args = _M.parse_flags(...)
    local addr = flags['addr'] or 'localhost'
    local port = flags['port'] or '8080'

    local fake = flags['test']
    local testing = flags['no_headers'] and fake
    last_obj = self

    if running then return
    else running = true
    end

    tracing = flags['trace']
    _M.tracing = tracing

    if flags['help'] then
        print(help_text)
        os.exit()
    end

    if fake then
        addr = fake==true and '/' or fake
    end

    if self.env then
        for k,v in pairs(self) do
            if type(v) == 'function' then
                setfenv(v,self.env)
            end
        end
    end

    -- create TCP socket on addr:port: allow for a debug hook
    local server_ctor = fake and require 'orbiter.fake' or socket_bind
    local server, port = assert(server_ctor(addr, tonumber(port)))
    if not fake then
        local URL = 'http://'..addr..':'..port
        print ("Orbiter serving on "..URL)
        if  flags['launch'] then
            launch_browser(URL,flags['launch'])
        end
    end
    -- loop while waiting for a user agent request
    while 1 do
        -- wait for a connection, set timeout and receive request from user agent
        local client = server:accept()
        client:settimeout(60)
        local request, err = client:receive()
        if not err then
            local content,file,action,captures,obj,web,vars,headers,err,sheaders,url
            if tracing then trace('request: '..request) end
            local method = request:match '^([A-Z]+)'
            if not fake then
                headers,err = receiveheaders(client)
            end
            if not err then
                if headers then
                    headers.HTTP_REMOTE = client:getpeername()
                end
                if method == 'POST' then
                    local size = tonumber(headers.HTTP_CONTENT_LENGTH)
                    vars = client:receive(size)
                    if tracing then trace('post: '..vars) end
                end
                -- resolve requested file from user agent request
                file = request:match('(/%S*)')
                if method == 'GET' then
                    url,vars = file:match('([^%?]+)%?(.+)')
                    if url then file = url end
                end
                vars = vars and url_split(vars) or {}
				-- setup the web object
				web = Web:new{ vars = headers, input = vars, method = method:lower(), path_info = file }
                web[method=='GET' and 'GET' or 'POST'] = vars

                file, obj = process_request_filters(web,file)
                action, captures, obj = match_patterns(method, file, obj)
                if action then
                    -- @doc handlers may specify the MIME type of what they
                    -- return, if they choose; default is HTML.
                    local status, content, mime = pcall(action, obj, web, unpack(captures))

                    if status then
                        if not content and method ~= 'POST' then
                            status = false
                            content = '404 Request Failed'
                        elseif mime == false then
                            status = false
                        end
                    else
                        print('exception: '..tostring(content))
                    end
                    if content then -- can naturally be nil for POST requests!
                        if status then
                            -- @doc if the app or extension object defines a content_filter method,
                            -- it will receive the content and mime type, and is expected to
                            -- return the same.
                            if self.content_filter then
                                content, mime = self:content_filter(content, mime)
							end
                            if not testing then
                                send_headers(client, web.status, mime or 'text/html', #content, web.headers)
                            end
                            client:send(content)
                        else
                            send_error(client, content)
                        end
                    end
                else -- unmatched pattern!!
                    send_error (client,'404 Not Found')
                end
            else
                print ('header parsing failed:',err)
            end
        else
            print ('client receive failed',err)
        end
        -- done with client, close request
        client:close()
    end
end

return _M  -- orbiter!
