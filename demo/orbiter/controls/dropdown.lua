-- A simple drop-down menu
-- http://javascript-array.com/scripts/simple_drop_down_menu/

local html=require 'orbiter.html'
local _M = {}

html.set_defaults {
    inline_style = [[

#sddm
{	margin: 0;
	padding: 0;
	z-index: 30
}

#sddm li
{	margin: 0;
	padding: 0;
	list-style: none;
	float: left;
	font: bold 11px arial
}

#sddm li a
{	display: block;
	margin: 0 1px 0 0;
	padding: 4px 10px;
	width: 60px;
	background: #5970B2;
	color: #FFF;
	text-align: center;
	text-decoration: none
}

#sddm li a:hover
{	background: #49A3FF}

#sddm div
{	position: absolute;
	visibility: hidden;
	margin: 0;
	padding: 0;
	background: #EAEBD8;
	border: 1px solid #5970B2
}

	#sddm div a
	{	position: relative;
		display: block;
		margin: 0;
		padding: 5px 10px;
		width: auto;
		white-space: nowrap;
		text-align: left;
		text-decoration: none;
		background: #EAEBD8;
		color: #2875DE;
		font: 11px arial
	}

	#sddm div a:hover
	{	background: #49A3FF;
		color: #FFF
	}

]],
    inline_script = [[

<!--
var timeout         = 500;
var closetimer		= 0;
var ddmenuitem      = 0;

// open hidden layer
function mopen(id)
{
	// cancel close timer
	mcancelclosetime();

	// close old layer
	if(ddmenuitem) ddmenuitem.style.visibility = 'hidden';

	// get new layer and show it
	ddmenuitem = document.getElementById(id);
	ddmenuitem.style.visibility = 'visible';

}
// close showed layer
function mclose()
{
	if(ddmenuitem) ddmenuitem.style.visibility = 'hidden';
}

// go close timer
function mclosetime()
{
	closetimer = window.setTimeout(mclose, timeout);
}

// cancel close timer
function mcancelclosetime()
{
	if(closetimer)
	{
		window.clearTimeout(closetimer);
		closetimer = null;
	}
}

// close layer when click-out
document.onclick = mclose;
// -->
    ]],
}
local a,div = html.tags 'a,div'

local function item(label,idx,items)
    local id = "m"..idx
    local link = a { href='#',onmouseover="mopen('"..id.."')",onmouseout="mclosetime()", label}
    local idiv = div {id = id, onmouseover="mcancelclosetime()" ,onmouseout="mclosetime()"}
    local j = 1
    for i = 1,#items,2 do
        idiv[j] = html.link(items[i+1],items[i])
        j = j + 1
    end
    return { link, idiv }
end

function _M.menu(items)
    local ls = {}
    local j = 1
    for i = 1,#items,2 do
        ls[j] = item(items[i],j,items[i+1])
        j = j + 1
    end
    ls.id = 'sddm'
    return {html.list(ls),div {style='clear:both',''}}
end

return _M  -- orbiter.controls.dropdown
