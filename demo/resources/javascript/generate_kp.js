function initArray(length, value) {
    var arr = []; 
    for (var i = 0; i < length; i++) {
        arr.push(value);
    }
    return arr; 
}

function get_part_name(part_id) {
    switch (parseInt(part_id) - 1) {
        case 0: 
            return "Back";
            break;
        case 1:
            return "Beak";
            break;
        case 2: 
            return "Belly";
            break;
        case 3: 
            return "Breast";
            break;
        case 4: 
            return "Crown";
            break;
        case 5:
            return "Forehead";
            break;
        case 6: 
            return "Left Eye";
            break;
        case 7:
            return "Left Leg";
            break;
        case 8: 
            return "Left Wing";
            break;
        case 9: 
            return "Nape";
            break;
        case 10: 
            return "Right Eye";
            break;
        case 11: 
            return "Right Leg";
            break;
        case 12: 
            return "Right Wing";
            break;
        case 13: 
            return "Tail";
            break;
        case 14: 
            return "Throat";
            break;
    }
}

$(document).ready(function() {

    var mode = 'pp'; // default to bounding box
    var canvas, context, startX, endX, startY, endY;
    var mouseIsDown = 0;
    var option_selected = null; 

    var checklist = initArray(15, false);

    canvas = document.getElementById("pp_canvas");
    context = canvas.getContext("2d");
    var cw = canvas.width;
    var ch = canvas.height;

    canvas.addEventListener("mousedown", mouseDown_pp, false);
    canvas.addEventListener("mousemove", mouseXY_pp, false);
    canvas.addEventListener("mouseup", mouseUp_pp, false);

    function findSelection(field) {
        var test = document.getElementsByName(field);
        var sizes = test.length;
        console.log(sizes);
        for (var i=0; i < sizes; i++) {
            if (test[i].checked==true) {
                // alert(test[i].value + ' you got a value');     
                return test[i].value;
            }
        }
    }

    function submitForm(field_name) {

        var opt_select =  findSelection(field_name);
        return opt_select;
    }


    // Ajax requests
    function get_bb_prediction() {

        var description = $('#description').val();

        var obj = {
            x: startX, 
            y: startY, 
            width: Math.abs(endX - startX),
            height: Math.abs(endY - startY), 
            description: description
        }; 

        $.ajax({
            data: obj,
            url: 'request',
            type: "GET",
            success: function (msg) {
                console.log(msg)
                //img = new Image()
                //img.src = msg
                //context.drawImage(img, 0, 0, 500, 500)
                context.clearRect(0, 0, canvas.width, canvas.height);
                $('.placeholder').attr('src', msg)
                //$('#placeholder2').attr('src', msg)
            },
            error: function (msg) { ret = 'Epic fail!'; },
            async: false,
            timeout: 10000,
        });
    }
    
    function get_pp_prediction() {

        var description = $('#pp_description').val();
        var keypoints = []; 
        var canvas_obj = $("#pp_canvas");
        var num_canvas_objs = canvas_obj.children.length; 
        for (var i = 0; i < num_canvas_objs; i++) {

            var current_child = canvas_obj.children()[i];
            
            if (current_child == null) {
                continue; 
            }

            var x = current_child.x; 
            var y = current_child.y; 
            var part_id = current_child.part; 
            var keypoint_obj = {
                x: x,
                y: y,
                part_id: part_id,
            };

            keypoints.push(keypoint_obj);
            console.log(keypoints);
        }

        var obj = {
            description: description, 
            keypoints: keypoints
        };

        $.ajax({
            data: { "data" : JSON.stringify(obj) },
            url: 'request',
            type: "POST",
            success: function (msg) {
                console.log(msg)
                //img = new Image()
                //img.src = msg
                //context.drawImage(img, 0, 0, canvas.width, canvas.height);
                // context.clearRect(0, 0, canvas.width, canvas.height);
                $('.placeholder').attr('src', msg)
                //$('#placeholder2').attr('src', msg)
            },
            error: function (msg) { ret = 'Epic fail!'; },
            async: true,
            contentType: "application/json",
            timeout: 10000,
        });

        // document.getElementById('pp_description').value='';
        //checklist = initArray(15, false);
    }

    // canvas functions

    function mouseUp(eve) {
        if (mouseIsDown !== 0) {
            mouseIsDown = 0;

            // keep it on canvas
            context.beginPath(); 
            context.fillStyle = "rgb(255, 255, 255, 0.5)";
            context.fillRect(startX, startY, Math.abs(startX - endX), Math.abs(startY - endY));

            // create an object out of init
            var elt = document.createElement('div');
            elt.style.top = startY + 'px';
            elt.style.left = startX + 'px';
            elt.style.width = Math.abs(startX - endX) + 'px';
            elt.style.height = Math.abs(startY - endY) + 'px';
            canvas.removeChild(canvas.childNodes[0]); // remove if we want more than one bounding Box

            canvas.appendChild(elt); // put bounding box into DOM

            canvas.style.cursor = "default"; // reset cursor

            var pos = getMousePos(canvas, eve);
            endX = pos.x;
            endY = pos.y;
            drawSquare(); //update on mouse-up

        }
    }

    function mouseDown(eve) {
        mouseIsDown = 1;
        canvas.style.cursor = "crosshair";
        var pos = getMousePos(canvas, eve);
        startX = endX = pos.x;
        startY = endY = pos.y;
        drawSquare(); //update
    }

    function mouseXY(eve) {

        if (mouseIsDown !== 0) {
            var pos = getMousePos(canvas, eve);
            endX = pos.x;
            endY = pos.y;

            drawSquare();
        }
    }

    function drawSquare() {
        // creating a square
        var w = endX - startX;
        var h = endY - startY;
        var offsetX = (w < 0) ? w : 0;
        var offsetY = (h < 0) ? h : 0;
        var width = Math.abs(w);
        var height = Math.abs(h);

        context.clearRect(0, 0, canvas.width, canvas.height);
                   
        context.beginPath();
        context.fillRect(startX + offsetX, startY + offsetY, width, height);
        context.fillStyle = 'rgba(255, 0, 0, 0.5)';
        context.fill();
        context.lineWidth = 5;
        context.strokeStyle = 'black';
        context.stroke();
    }

    function drawKeyPoint() {

        var dialog = document.querySelector('dialog');
        dialogPolyfill.registerDialog(dialog);
        // Now dialog acts like a native <dialog>.
        dialog.showModal();

        // assumes a default value is chosen 
        $(".agree").click(function () {
            
            var part_selected = submitForm("part_select");

            // verify part has not been selected before
            if (checklist[parseInt(part_selected) - 1] == true) {
                return;
            } 
            else {
                checklist[parseInt(part_selected) - 1] = true; 
            }
        
            // draw keypoint          
            context.beginPath();
            context.fillRect(startX, startY, 10, 10);
            context.fillStyle = 'rgba(0, 0, 0, 1)';
            context.font = '15px Georgia';
            context.fillText(get_part_name(part_selected), startX, startY);
            context.fill();
            context.lineWidth = 5;
            context.strokeStyle = 'red';
            context.stroke();

            // create an object out of init
            var elt = document.createElement('div');
            elt.style.top = startY + 'px';
            elt.style.left = startX + 'px';

            elt.x = startX;
            elt.y = startY;
            elt.part = part_selected;
            console.log("element part: " + elt.part); 
            canvas.appendChild(elt); // put bounding box into DOM
            dialog.close(); 

            return; 
        });

        $(".close").click(function () {
            dialog.close();
            return null; 
        });
    }


    function handle_keypoints() {
        var keypoint_args = drawKeyPoint(); 
        canvas.style.cursor = "default"; // reset cursor
    }

    function getMousePos(canvas, evt) {
        var rect = canvas.getBoundingClientRect();
        return {
            x: evt.clientX - rect.left,
            y: evt.clientY - rect.top
        };
    }

    // Pick part helpers
    function mouseDown_pp(eve) {
        mouseIsDown = 1;
        canvas.style.cursor = "crosshair";
        var pos = getMousePos(canvas, eve);
        startX = endX = pos.x;
        startY = endY = pos.y;
        context.beginPath();
        context.fillRect(startX, startY, 1, 1);
        context.fillStyle = 'rgba(255, 0, 0, 0.5)';
        context.fill();
        context.lineWidth = 5;
        context.strokeStyle = 'red';
        context.stroke();

        // handle_keypoints(); 
    }


    function mouseUp_pp(eve) {
        if (mouseIsDown !== 0) {
            mouseIsDown = 0;
            handle_keypoints();             
        }
    }

    function mouseXY_pp(eve) {

        if (mouseIsDown !== 0) {
            var pos = getMousePos(canvas, eve);
            endX = pos.x;
            endY = pos.y;
            // handle_keypoints(); 
            // var keyPoint_args = drawKeyPoint(); 
        }
    }


    // Event listeners

    $('#generate_btn').click(function () {
        get_bb_prediction(); 
    });

    $('#pp_generate_btn').click(function () {
        get_pp_prediction(); 
    });

    // When enter is pressed, execute prediction
    document.getElementById("pp_description").addEventListener("keydown", function(e) {
        if (!e) { var e = window.event; }

        // Enter is pressed
        if (e.keyCode == 13) { 
            e.preventDefault(); // sometimes useful
            get_pp_prediction(); 
        }
    }, false);

    $("#pp_clear_btn").click(function () {
        var canvas = document.getElementById("pp_canvas");
        var context = canvas.getContext("2d");
        context.clearRect(0, 0, canvas.width, canvas.height);
        checklist = initArray(15, false);
        var canvas = $("#pp_canvas");
        canvas.empty();
        document.getElementById('pp_description').value='';
    });
});
