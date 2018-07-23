/*
variables
*/
var model;
var canvas;
var classNames = [];
var canvas;
var coords = [];
var mousePressed = false;

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
        getFrame();
        mousePressed = false
    });
    canvas.on('mouse:down', function(e) {
        mousePressed = true
    });
    canvas.on('mouse:move', function(e) {
        recordCoor(e)
    });
}

/*
record the current drawing coordinates
*/
function recordCoor(event) {
    var pointer = canvas.getPointer(event.e);
    var posX = pointer.x;
    var posY = pointer.y;

    if (posX >= 0 && posY >= 0 && mousePressed) {
        coords.push(pointer)
    }
}

/*
get the best bounding box by trimming around the drawing
*/
function getMinBox() {
    //get coordinates 
    var coorX = coords.map(function(p) {
        return p.x
    });
    var coorY = coords.map(function(p) {
        return p.y
    });

    //find top left and bottom right corners 
    var min_coords = {
        x: Math.min.apply(null, coorX),
        y: Math.min.apply(null, coorY)
    }
    var max_coords = {
        x: Math.max.apply(null, coorX),
        y: Math.max.apply(null, coorY)
    }

    //return as strucut 
    return {
        min: min_coords,
        max: max_coords
    }
}

/*
get the current image data 
*/
function getImageData() {
    //get the minimum bounding box around the drawing 
    const mbb = getMinBox()

    //get image data according to dpi 
    const dpi = window.devicePixelRatio
    const margin = 2 
    
    const x = (mbb.min.x - margin) * dpi 
    const y = (mbb.min.y - margin) * dpi
    const w = (mbb.max.x - mbb.min.x + 2 * margin) * dpi 
    const h = (mbb.max.y - mbb.min.y + 2 * margin) * dpi 
    console.log(x +' '+ y +' '+w+' '+h+' ')
    const imgData = canvas.contextContainer.getImageData(x, y, w, h)
    return imgData
}

/*
get the prediction 
*/
function getFrame() {
    //make sure we have at least two recorded coordinates 
    if (coords.length >= 2) {
        
        //get the image data from the canvas 
        const imgData = getImageData();

        //get the prediction 
        const gImg = model.predict(preprocess(imgData))
        
        //draw on canvas 
        const gCanvas = document.getElementById('gCanvas');
        const postImg = postprocess(gImg)
        tf.toPixels(postImg, gCanvas)
    }

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
     return tf.tidy(() => {
        //normalization factor  
        const scale = tf.scalar(0.5);
        
        //unnormalize and sqeeze 
        const squeezed = tensor.squeeze().mul(scale).add(scale)

        //resize to canvas size 
        let resized = tf.image.resizeBilinear(squeezed, [300, 300])
        return resized
    })
}

/*
load the model
*/
async function start() {
    //load the model 
    model = await tf.loadModel('shoes/model.json')
    
    //status 
    document.getElementById('status').innerHTML = 'Model Loaded';
    
    //warm up 
    model.predict(tf.zeros([1, 256, 256, 3]))
    
    //allow drawing on the canvas 
    allowDrawing()
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
    var slider = document.getElementById('myRange');
    slider.oninput = function() {
        canvas.freeDrawingBrush.width = this.value;
    };
}

/*
clear the canvas 
*/
function erase() {
    canvas.clear();
    canvas.backgroundColor = '#ffffff';
    coords = [];
}

//start the script 
 $(window).on('load', function(){
    prepareCanvas();
    start();
 });
