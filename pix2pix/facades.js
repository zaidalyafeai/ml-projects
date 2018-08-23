/*
variables
*/
var model;
var canvas;
var currColor = '#002FFF'
var backColor = '#0000DE'
/*
color pallette click events
*/
$(document).on("click","td", function(e){
    //get the color 
    const color = e.target.style.backgroundColor;
    //set the color 
    currColor = color;
});

/*
prepare the drawing canvas 
*/
function prepareCanvas() {
    canvas = window._canvas = new fabric.Canvas('canvas');
    canvas.backgroundColor = backColor;
    canvas.renderAll();
    //setup listeners 
    canvas.observe('mouse:down', function(e) { mousedown(e); });
    canvas.observe('mouse:move', function(e) { mousemove(e); });
    canvas.observe('mouse:up', function(e) { mouseup(e); });

}

var started = false;
var x = 0;
var y = 0;

/* Mousedown */
function mousedown(e) {
    var mouse = canvas.getPointer(e);
    started = true;
    x = mouse.x;
    y = mouse.y;

    var square = new fabric.Rect({ 
        width: 0, 
        height: 0, 
        left: x, 
        top: y, 
        fill: currColor
    });

    canvas.add(square); 
    canvas.renderAll();
    canvas.setActiveObject(square); 
    square.set({selectable:false, hasControls:false, hasBorders:false})
}


/* Mousemove */
function mousemove(e) {
    if(!started) {
        return false;
    }

    var mouse = canvas.getPointer(e);

    var w = Math.abs(mouse.x - x),
    h = Math.abs(mouse.y - y);

    if (!w || !h) {
        return false;
    }

    var square = canvas.getActiveObject(); 
    square.set('width', w).set('height', h);
    canvas.renderAll(); 
}

/* Mouseup */
function mouseup(e) {
    if(started) {
        started = false;
    }

    var square = canvas.getActiveObject();
    canvas.add(square); 
    square.set({hasControls:false, hasBorders:false})
    square.evented = false
    canvas.renderAll();
    const imgData = getImageData();
    predict(imgData)
 } 

/*
get the current image data 
*/
function getImageData() {
    //get image data according to dpi 
    const dpi = window.devicePixelRatio    
    const x = 0 * dpi 
    const y = 0 * dpi
    const w = canvas.width * dpi 
    const h = canvas.height * dpi 
    const imgData = canvas.contextContainer.getImageData(x, y, w, h)
    return imgData
}

/*
get the prediction 
*/
function predict(imgData) {

    //get the prediction 
    const gImg = model.predict(preprocess(imgData))

    //draw on canvas 
    const gCanvas = document.getElementById('gCanvas');
    const postImg = postprocess(gImg)
    tf.toPixels(postImg, gCanvas)
}

/*
preprocess the data
*/
function preprocess(imgData) {
    return tf.tidy(() => {
        //convert to a tensor 
        let tensor = tf.fromPixels(imgData).toFloat()
        //resize 
        let resized = tf.image.resizeBilinear(tensor, [256, 256])
                
        //normalize 
        const offset = tf.scalar(127.5);
        const normalized = resized.div(offset).sub(tf.scalar(1.0));

        //We add a dimension to get a batch shape 
        const batched = normalized.expandDims(0)
        
        return batched
    })
}

/*
post process 
*/
function postprocess(tensor){
     const w = canvas.width  
     const h = canvas.height 
     
     return tf.tidy(() => {
        //normalization factor  
        const scale = tf.scalar(0.5);
        
        //unnormalize and sqeeze 
        const squeezed = tensor.squeeze().mul(scale).add(scale)

        //resize to canvas size 
        let resized = tf.image.resizeBilinear(squeezed, [w, h])
        return resized
    })
}

function populateInitImage()
{
    var imgData = new Image;
    imgData.src = "facade.png"
    imgData.onload = function () {
        var img = new fabric.Image(imgData, {
            width: 256,
            height: 256,
        });
        canvas.add(img)
        img.evented = false
        canvas.renderAll();
        predict(imgData)
    }
}

/*
load the model
*/
async function start() {
    //load the model 
    model = await tf.loadModel('facades_model/model.json')
    
    //status 
    document.getElementById('status').innerHTML = 'Model Loaded';
    
    //warm up 
    populateInitImage()
    
    $('button').prop('disabled', false);
}

/*
clear the canvas 
*/
function erase() {
    canvas.clear();
    canvas.backgroundColor = backColor;
}

//start the script 
 $(window).on('load', function(){
    prepareCanvas();
    start();
 });
