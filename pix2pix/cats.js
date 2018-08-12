/*
variables
*/
var model;
var canvas;
var currColor = '#002FFF'
var backColor = '#ffffff'

/*
slider
*/
var max = 10,
    min = 1,
    step = 1,
    output = $('#output').text(min);

$("#range-slider")
    .attr({'max': max, 'min':min, 'step': step,'value': String(min)})
    .on('input change', function() {
    
        output.text(this.value);
});

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
    canvas.backgroundColor = '#ffffff';
    canvas.isDrawingMode = 1;
    canvas.freeDrawingBrush.color = "black";
    canvas.freeDrawingBrush.width = 1;
    canvas.renderAll();
    //setup listeners 
    canvas.on('mouse:up', function(e) {
        const imgData = getImageData();
        predict(imgData)
        mousePressed = false
    });
    canvas.on('mouse:down', function(e) {
        mousePressed = true
    });
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
    imgData.src = "cat.png"
    imgData.onload = function () {
        var img = new fabric.Image(imgData, {
            width: 256,
            height: 256,
        });
        canvas.add(img)
        predict(imgData)
    }
}

/*
load the model
*/
async function start() {
    //load the model 
    model = await tf.loadModel('cats_model/model.json');
    
    //status 
    document.getElementById('status').innerHTML = 'Model Loaded';
    
    //warm up 
    populateInitImage();
    
    allowDrawing();
}

/*
allow drawing on canvas
*/
function allowDrawing() {
    //allow draing 
    canvas.isDrawingMode = 1;
    
    //alow UI 
    $('button').prop('disabled', false);
    
    //setup slider 
    var slider = document.getElementById('range-slider');
    slider.oninput = function() {
        canvas.freeDrawingBrush.width = this.value;
    };
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
