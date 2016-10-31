-- orbiter.fake
-- optional testing functionality - this returns a fake server constructor,
-- which just pumps our request into the loop, receives the output, and closes.
-- currently just does GET requests and is hard-wired to write to stdout.
-- The reader is invited to contemplate the difficulties of doing this in Java or C++.

local function fake_new (out,lines)
    local client = {}
    function client:receive()
        return table.remove(lines,1)
    end
    function client:send (stuff)
        out:write(stuff)
        return true
    end
    function client:settimeout ()
    end
    function client:close ()
        out:write '\n'
        os.exit()
    end
    return client
end

return function(url,port)
    local server = {}
    function server:accept()
        return fake_new (io.stdout, {
         "GET "..url.." HTTP/1.1\r\n",
        "\r\n"
        })
    end
    return server
end
