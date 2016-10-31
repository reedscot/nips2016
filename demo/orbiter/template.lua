-- Orbiter, a personal web application framework
-- Lua template preprocessor
-- Originally by Ricki Lake.
--

local append,format = table.insert,string.format

local parse1,parse2 = "()","(%b())()"
local parse_dollar

local load = load

if _VERSION:match '5%.1$' then -- Lua 5.1 compatibility
    function load(str,name,mode,env)
        local chunk,err = loadstring(str,name)
        if chunk then setfenv(chunk,env) end
        return chunk,err
    end
end

local function parseDollarParen(pieces, chunk, s, e)
    local s = 1
    for term, executed, e in chunk:gmatch (parse_dollar) do
        append(pieces,
            format("%q..(%s or '')..",chunk:sub(s, term - 1), executed))
        s = e
    end
    append(pieces, format("%q", chunk:sub(s)))
end
-------------------------------------------------------------------------------
local function parseHashLines(chunk)
    local find = string.find
    local pieces, s, args = find(chunk,"^\n*#ARGS%s*(%b())[ \t]*\n")
    if not args or find(args, "^%(%s*%)$") then
        pieces, s = {"return function(_put) ", n = 1}, s or 1
    else
        pieces = {"return function(_put, ", args:sub(2), n = 2}
    end
    while true do
        local ss, e, lua = find (chunk,"^#+([^\n]*\n?)", s)
        if not e then
            ss, e, lua = find(chunk,"\n#+([^\n]*\n?)", s)
            append(pieces, "_put(")
            parseDollarParen(pieces, chunk:sub(s, ss))
            append(pieces, ")")
            if not e then break end
        end
        append(pieces, lua)
        s = e + 1
    end
    append(pieces, " end")
    return table.concat(pieces)
end

local template = {}

function template.substitute(str,env)
    env = env or {}
    if env.__parent then
        setmetatable(env,{__index = env.__parent})
    end
    local out,res = {}
    parse_dollar = parse1..(env.__dollar or '$')..parse2
    local code = parseHashLines(str)
    local fn,err = load(code,'TMP','t',env)
    if not fn then return nil,err end
    res,fn = pcall(fn)
    if not res then return nil,err end
    res,err = pcall(fn,function(s)
        out[#out+1] = s
    end)
    if not res then return nil,err end
    return table.concat(out)
end

local cache = {}

function template.page(file,env)
    local app = env.app
    if app then file = app.resources..'/'..file end
    local do_cache = env.cache
    do_cache = do_cache==nil and true
    env.app = nil
    env.cache = nil
    local tmpl
    if do_cache then tmpl = cache[file] end
    if not tmpl then
        local f,err = io.open(file)
        if not f then return nil, err end
        tmpl = f:read '*a'
        f:close()
    end
    if do_cache then cache[file] = tmpl end
    return template.substitute(tmpl,env)
end

function template.templater (default_env)
    return function(file,env)
        for k,v in pairs(default_env) do env[k] = v end
        return template.page(file,env)
    end
end

return template




