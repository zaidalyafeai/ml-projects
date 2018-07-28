/*
variables
*/
var model;
var canvas;

/*
get the current image data 
*/
function getImageData() {
    //get image data according to dpi 
    var canvas = document.getElementById("canvas");
    var ctx = canvas.getContext("2d");
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return imgData
}

/*
get the prediction 
*/
function getFrame() {
    //get the image data from the canvas 
    const imgData = getImageData();
    
    //get the prediction 
    const gImg = model.predict(preprocess(imgData))
    
    //draw on canvas 
    const gCanvas = document.getElementById('gCanvas');
    tf.toPixels(postprocess(gImg), gCanvas)
}

/*
preprocess the data
*/
function preprocess(imgData) {
    return tf.tidy(() => {
        //convert to a tensor 
        let tensor = tf.zeros([256, 256, 3]).mul(tf.scalar(255)).toFloat()

        //normalize 
        const offset = tf.scalar(127.5);
        const normalized = tensor.div(offset).sub(tf.scalar(1.0))
        
        //We add a dimension to get a batch shape 
        const batched = normalized.expandDims(0)
        
        return batched
    })
}

/*
post process 
*/
function postprocess(tensor){
    tf.mean(tensor).print()
    const scale = tf.scalar(0.5);
    const unormalized = tensor.squeeze().mul(scale).add(scale)
    return unormalized
}

/*
load the model
*/
async function start() {
    //load the model 
    model = await tf.loadModel('model2/model.json')
    layers = model.layers;
    let tensor = tf.zeros([256, 256, 3]).mul(tf.scalar(255)).toFloat()

    //normalize 
    const offset = tf.scalar(127.5);
    const normalized = tensor.div(offset).sub(tf.scalar(1.0))
    
    //We add a dimension to get a batch shape 
    const batched = normalized.expandDims(0)

    const model2 = tf.model({
        inputs: layers[0].input, 
        outputs: layers[0].output
    })

    model2.predict(batched).print()
    //model.layers[1].getWeights()[0].print()
    //status 
    document.getElementById('status').innerHTML = 'Model Loaded';

    //warm up 
    model.predict(tf.zeros([1, 256, 256, 3]))
    $('button').prop('disabled', false);
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
clear the canvs 
*/
function erase() {
    getFrame()

}

//start the script 
$(window).on('load', function(){ 
    var c = document.getElementById("canvas");
    var ctx = c.getContext("2d");
    var img = new Image;
    img.src = "cat.jpg"
    img.onload = function () {
    ctx.drawImage(img, 0, 0, c.width, c.height);
    start()
    }
});
