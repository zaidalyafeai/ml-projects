async function loadModel()
{
  model = await tf.loadModel('model/model.json')
  model.predict(tf.zeros([1, 2, 2, 3])).print()
}

loadModel()
