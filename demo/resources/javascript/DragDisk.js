function DragDisk(posX, posY, text, id) {
    this.id = id
	this.x = posX;
	this.y = posY;
    this.radius = 15;
    this.text = text;
    this.bigcolor = 'rgba(255,0,0,0.8)';
    this.smallcolor = 'rgba(255,255,255,0.5)';
    this.textcolor = 'rgba(0,0,0,1)';
}

/*Checks whether given point lies inside disk.*/
DragDisk.prototype.hitTest = function(hitX,hitY) {
	var dx = this.x - hitX;
	var dy = this.y - hitY;
	return(dx*dx + dy*dy < this.radius*this.radius);
}

/*Draws disk.*/
function draw_disk(y, x, r, color, context) {
    context.fillStyle = color;
	context.beginPath();
	context.arc(x, y, r, 0, 2*Math.PI, false);
	context.closePath();
    context.fill();
}
/*Draws labeled concentric disks.*/
DragDisk.prototype.drawToContext = function(theContext) {
    /*window.alert("!" + this.y + '!' + this.x);*/
    draw_disk(this.y, this.x, this.radius*1.0, this.bigcolor, theContext);
    draw_disk(this.y, this.x, this.radius*0.5, this.smallcolor, theContext);

    theContext.textAlign='center';
    theContext.textBaseline='middle';
    theContext.font = '700 11px Georgia';
    theContext.fillStyle = this.textcolor;
    theContext.fillText(this.text, this.x, this.y);
}
