$(document).ready(function() {

    var canvas, context, startX, endX, startY, endY;
    var mouseIsDown = 0;

    function init() {
        canvas = document.getElementById("draw_box_canvas");
        context = canvas.getContext("2d");

        canvas.addEventListener("mousedown", mouseDown, false);
        canvas.addEventListener("mousemove", mouseXY, false);
        canvas.addEventListener("mouseup", mouseUp, false);
    }

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

    function getMousePos(canvas, evt) {
        var rect = canvas.getBoundingClientRect();
        return {
            x: evt.clientX - rect.left,
            y: evt.clientY - rect.top
        };
    }
    init();

    // document.getElementById("generate_btn").addEventListener("click", function() {
    //     document.getElementById('obj_img').innerHTML = "<img src='resources/images/result.jpg'>";
    // });

    $('#generate_btn').click(function () {

        var description = $('#description').val();

        var obj = {
            x: startX, 
            y: startY, 
            width: Math.abs(endX - startX),
            height: Math.abs(endY - startY), 
            description: description
        }
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
                $('#placeholder').attr('src', msg)
                //$('#placeholder2').attr('src', msg)
            },
            error: function (msg) { ret = 'Epic fail!'; },
            async: false,
            timeout: 10000,
        });

    });
});
