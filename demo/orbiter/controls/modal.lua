local jq = require 'orbiter.libs.jquery'
local html = require 'orbiter.html'
local form = require 'orbiter.form'

local _M = {}

html.set_defaults {
    inline_style = [[
#modalbox {
    position: absolute;
    left: 200px;
    top: 200px;
    background-color: #EEEEFF;
    border: 2px solid #000099;
}
#modaltitle {
    background-color: #000099;
    color: #EEEEFF;
}
#buttonrow {
    width: 100%;
    background-color: #EEEEFF;
}
]];

    inline_script = [[
function jq_close_modal() {
    $('#modalbox').remove()
}
]];
}

local div,button  = html.tags 'div,button'

local function _show_modal(f,web)
    f:prepare(web)
    -- we embed the form in a modal box
    return jq 'body' : append (
        div { id='modalbox';
        div { id='modaltitle', "Modal Dialog"},
        f:show();
        html.table {id = 'buttonrow'; {
          button{"OK",onclick=("jq_submit_form('%s'); jq_close_modal();"):format(f.id)},
          button{"Cancel",onclick="jq_close_modal();"},
        }}
    })
end

function _M.new (spec)
    spec.buttons = {}
    local f = form.new(spec)
    f.show_modal = _show_modal
    return f
end



return _M
