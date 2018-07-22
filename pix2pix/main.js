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
    const margin = 50 
    console.log(margin)
    const imgData = canvas.contextContainer.getImageData((mbb.min.x - margin) * dpi  , (mbb.min.y - margin) * dpi,
                                                        (mbb.max.x - mbb.min.x + margin) * dpi ,
                                                        (mbb.max.y - mbb.min.y + margin) * dpi );
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
        toImage(postImg, gCanvas)
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
    model = await tf.loadModel('model/model.json')
    
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

function toImage(tensor, canvas) {
    const ctx = canvas.getContext('2d');
    //get the tensor shape
    const [height, width] = tensor.shape;
    //create a buffer array
    const buffer = new Uint8ClampedArray(width * height * 4)
    //create an Image data var 
    const imageData = new ImageData(width, height);
    //get the tensor values as data
    const data = tensor.dataSync();
    //map the values to the buffer
    var i = 0;
    for(var y = 0; y < height; y++) {
    for(var x = 0; x < width; x++) {
        var pos = (y * width + x) * 4;      // position in buffer based on x and y
        buffer[pos  ] = Math.floor(data[i] * 255)             // some R value [0, 255]
        buffer[pos+1] = Math.floor(data[i+1] * 255)           // some G value
        buffer[pos+2] = Math.floor(data[i+2] * 255)           // some B value
        buffer[pos+3] = 255;                // set alpha channel
        i+=3
    }
  }
    //set the buffer to the image data
    imageData.data.set(buffer)
    //show the image on canvas
    ctx.putImageData(imageData, 0, 0, 0, 0, 300, 300);
  };
/*
clear the canvs 
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
