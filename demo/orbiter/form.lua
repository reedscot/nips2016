-- Orbiter, a personal web application framework
local orbiter = require 'orbiter'
local html = require 'orbiter.html'
local util = require 'orbiter.util'

local form = {}

local button_name = '-button'

html.set_defaults {
    inline_style = [[
  form.orbiter  fieldset { border: solid 1px #777; }
  form.orbiter table { border: solid 1px #000; }
  form.orbiter table td { padding: 6px; vertical-align:top; text-align: left }
  form.orbiter ul { list-style-type: none; padding: 0; margin: 0}
  form.orbiter ul li { padding-top: 6px; padding-bottom: 6px; }
]]
}

-- constraints

function form.range(x1,x2)
    return function(x)
        if x < x1 or x > x2 then return false,'must be between %f and %f' % {x1,x2}
        else return x end
    end
end

function form.match(pat,err)
    return function(s)
        if not s:find(pat) then return false,err else return s end
    end
end

form.non_blank = form.match('%S', 'may not be blank')

function form.irange(n1,n2)
    return function(x)
        if x < n1 or x > n2 then return false,'must be between %d and %d' % {n1,n2}
        elseif math.floor(x) ~= x then return false,'must be an integer'
        else return x
        end
    end
end

local converters = {
    number = {
        tostring = tostring,
        parse = function(s)
            local res = tonumber(s)
            if not res then return false,'not a number'
            else return res
            end
        end
    },
    boolean = {
        tostring = tostring,
        parse = function(s) return s=='true' and true or false end
    },
    string = {
        tostring = tostring,
        parse = tostring,
    }
}

local select_,option_,form_,label_,textarea_,fieldset_,legend_ = html.tags 'select,option,form,label,textarea,fieldset,legend'
local p = html.tags 'p'

local function value_set(c,v) c:set_attrib('value',v) end
local function value_get(c) return c:get_attrib('value') end
local function child1_set(c,v) c[1] = v end
local function child1_get(c) return c[1] end


local function input(type,name,value,size)
    local ctrl = html.elem('input',{
        type=type,name=name,value=value,size=tostring(size)
    })
    ctrl.set = value_set
    ctrl.get = value_get
    return ctrl
end

local Constraint = util.class()()

form.textarea = util.class(Constraint) {
    init = function(self,args)
        self.rows = args.rows
        self.cols = args.cols
    end;
    converter = converters.string,
    control = function(self,name,value)
        local ctrl = textarea_{name=name,
            rows=tostring(self.rows),cols=tostring(self.cols);
            value}
            ctrl.set = child1_set
            ctrl.get = child1_get
            return ctrl
    end,
    __tostring = function(self) return 'text area' end
}

local function text(name,value,size)
    return input('text',name,value,size)
end

local function checkbox(name,value,size)
    return input('checkbox',name,value)
end

local function listbox(name,value,list,size,multiple)
    local ctrl = select_ {name=name,
        size = (size and tostring(size) or nil),
        multiple = (multiple and 'multiple' or nil)
    }
    for i,v in ipairs(list) do
        ctrl[i] = option_{value=v,selected = (v==value and 'selected' or nil),v}
    end
    ctrl.set = value_set
    ctrl.get = value_get
    return ctrl
end

local function generate_control(obj,var,constraint)
    local value = obj[var]
    local vtype = type(value)
    local cntrl
    local converter = converters.string
    --print(var,constraint,value,vtype)
    if util.class_of(constraint,Constraint) then
        --print('value',value)
        cntrl = constraint:control(var,value)
        converter = constraint.converter
        constraint = constraint.constraint
    else
        if vtype == 'number' then
            converter = converters.number
            cntrl = text(var,converter.tostring(value),'8')
        elseif vtype == 'boolean' then
            converter = converters.boolean
            cntrl = checkbox(var,converter.tostring(value))
        elseif vtype == 'string' then
            local size
            if type(constraint) == 'number' then
                size = constraint
                constraint = nil
            end
            if util.is_plain(constraint) then
                local data = constraint
                if not util.is_list(data) then
                    data = constraint.data
                end
                if util.is_list(data) then
                    cntrl = listbox(var,value,data,constraint.size,constraint.multiple)
                    constraint = nil
                end
            else
                cntrl = text(var,value,size)
            end
        end
    end
    return cntrl,converter,constraint
end

function form.new (t)
    local f = { spec_of = {}, spec_table = t }
    f.validate = form.validate
    f.show = form.show
    f.prepare = form.prepare
    f.create = form.create
    return f
end

local append = table.insert
local K = 1

function form.create (self,web)
    local spec = self.spec_table
    spec.action = spec.action or web.path_info
    if not spec.name then -- forms must have unique ids
        spec.name = 'form'..K
        K = K + 1
    end
    local obj = spec.obj
    self.obj = obj
    local res = {}
    for i = 1,#spec,3 do
        -- each row has these three values
        local label,var,constraint = spec[i],spec[i+1],spec[i+2]
        local id = var -- might be better to autogenerate to ensure uniqueness?
        local cntrl,converter,constraint = generate_control(obj,var,constraint)
        cntrl:set_attrib('id',id)
        label = label_{['for']=id,title=label,label}
        local spec = {label=label,cntrl=cntrl,converter=converter,constraint=constraint}
        append(res,spec)
        self.spec_of[var] = spec
    end
    self.spec_list = res
    local contents
    local ftype = spec.type or 'cols'
    if ftype == 'cols' or ftype == 'rows' then -- wrap up as a table
        local tbl = {}
        for i,item in ipairs(res) do
            tbl[i] = {item.label,item.cntrl}
        end
        if ftype == 'rows' then tbl = util.transpose(tbl) end
        contents = html.table{  data = tbl }
    elseif spec.type == 'list' or spec.type == 'line' then
        local items = {}
        for i,item in ipairs(res) do
            append(items,item.label)
            append(items,item.cntrl)
        end
        if spec.type == 'list' then
            contents = html.list { data = items }
        else
            contents = p(items)
        end
    end
    if spec.title then
        contents = fieldset_{legend_(spec.title),contents}
    end
    self.id = spec.name
    local action, obj = spec.action, self.obj
    if obj.dispatch_post and type(action) == 'function' then
        local callback = action
        action = orbiter.prepend_root('/form/'..self.id)
        obj:dispatch_post(function(app,web)
            self:prepare(web)
            return callback(app,web,self)
        end,action)
    end
    self.body = form_{
        name = spec.name; id = self.id; action = action, class = 'orbiter';
        method = spec.method or 'post';
        contents,
    }
    spec.buttons = spec.buttons or {'submit'}
    for _,b in ipairs(spec.buttons) do
        append(self.body,input('submit',button_name,b))
    end
    return self.body
end

function form.prepare(self,web)
    if web.method == 'get' then
        self:create(web)
        return true
    else -- must be 'post'
        local ok,resp = self:validate(web.input)
        return not ok
    end
end

function form.show(self)
    return self.body
end

function form.validate (self,values)
    local ok = true
    local res = {}
    self.button = values[button_name]
    for var,value in pairs(values) do
        local spec = self.spec_of[var]
        if spec then
            spec.cntrl:set(value)
            local val,err = spec.converter.parse(value)
            if val and spec.constraint then
                val,err = spec.constraint(val)
            end
            if err then
                ok = false
                spec.cntrl:set_attribs {
                    style='background:pink',
                    title = err
                }
            else
                res[var] = val
            end
        end
    end
    if not ok then
        return false
    else -- only do this if everything is fine!
        util.update(self.obj,res)
        return true
    end
end

return form
