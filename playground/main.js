async loadModel()
{
  model = tf.loadMode('model/model.json')
  model.predict(tf.zeros(1, 2, 2, 3)).print9)
}

loadModel()
