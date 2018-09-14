/*
variables
*/
var model = undefined;
var canvas;
var backColor = 'white'
var gCanvas = document.getElementById('gCanvas');

/*
slider
*/
$("#range-slider")
    .on('input change', function() {
        $('#output').text(this.value);
        canvas.freeDrawingBrush.width = parseInt(this.value);
});


/*
color pallette click events
*/
$(document).on("click","td", function(e){
    //get the color 
    const color = e.target.style.backgroundColor;
    //set the color 
    canvas.freeDrawingBrush.color = color;
});

/*
prepare the drawing canvas 
*/
function prepareCanvas() {
    canvas = window._canvas = new fabric.Canvas('canvas');
    canvas.backgroundColor = backColor;
    canvas.isDrawingMode = 1;
    canvas.freeDrawingBrush.color = backColor;
    canvas.freeDrawingBrush.width = parseInt($("#output").text());
    canvas.renderAll();
    //setup listeners 
    canvas.on('mouse:up', function(e) {
        const imgData = getImageData();
        const pred = predict(imgData)
        tf.toPixels(pred, gCanvas)

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
    const x = 0 
    const y = 0 
    const w = canvas.width * dpi 
    const h = canvas.height * dpi 
    const imgData = canvas.contextContainer.getImageData(x, y, w, h)
    return imgData
}

/*
get the prediction 
*/
function predict(imgData) {
    return tf.tidy(() => {
        //get the prediction 
        const gImg = model.predict(preprocess(imgData))
        //post process 
        const postImg = postprocess(gImg)
        return postImg   
    })
}

/*
preprocess the data
*/
function preprocess(imgData) {
    return tf.tidy(() => {
        //convert to a tensor 
        const tensor = tf.fromPixels(imgData).toFloat()

        //resize 
        const resized = tf.image.resizeBilinear(tensor, [256, 256])
                
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
     const w = gCanvas.width  
     const h = gCanvas.height 
     
     return tf.tidy(() => {
        //normalization factor  
        const scale = tf.scalar(0.5);
        
        //unnormalize and sqeeze 
        const squeezed = tensor.squeeze().mul(scale).add(scale)

        //resize to canvas size 
        const resized = tf.image.resizeBilinear(squeezed, [w, h])
        return resized
    })
}

/*
predict on initial image
*/
function populateInitImage(imgName)
{
    var imgData = new Image;
    imgData.src = imgName
    imgData.onload = function () {
        const img = new fabric.Image(imgData, {
            scaleX: canvas.width / 256,
            scaleY: canvas.height / 256,
        });
        canvas.add(img)
        const pred = predict(imgData)
        tf.toPixels(pred, gCanvas)
    }
}

/*
load the model
*/
async function start(imgName, modelPath) {
    //load the model 
    model = await tf.loadModel(modelPath);
    
    //status 
    document.getElementById('status').innerHTML = 'Model Loaded';
    document.getElementById('bar').style.display = "none"
    //warm up 
    populateInitImage(imgName);
    
    allowDrawing();
}

/*
allow drawing on canvas
*/
function allowDrawing() {
    //allow draing 
    canvas.isDrawingMode = 1;
    
    //alow UI 
    $('#clear').prop('disabled', false);
    
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

/*
release resources when leaving the current page
*/
function release()
{
    if(model != undefined)
    {
        model.dispose()
    }
}
window.onbeforeunload = function (e) {
    console.log('leaving the page')
    release()
}
$('.nav-link').click(function ()
{
    release()
})

