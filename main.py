import tensorflow as tf
from LFWDataLoader import LFWDataLoader
import sys

if len(sys.argv) != 2:
  print("Provide the location for your dataset")
  sys.exit()


# Initialize parameters
data_path = sys.argv[1]
epochs = 500

# Location of pretrained CNN ProtoBuf
# pretrained_graph_path = "InceptionResnetV2.pb"
pretrained_graph_path = "InceptionV3.pb"


def triplet_loss_model(alpha=0.2):
  """
  Assemble tensorflow model that trains last several layers of CNN to enable face recognition
  :param alpha: Margin value for triplet loss
  :return: dictionary with model tensors:
  - loss
  - train: tensor for updating weights
  - anchors: placeholder for achors
  - positives: placeholder for positive examples
  - negatives: placeholder for negative examples
  - accuracy: accuracy metric
  - lr: learning rate supplied to optimized. Enables learning rate annealing
  - thr_ckh: returns classificaiton decisions for a hardcoded threshold
  - sq_norms: useful for fitting decision threshold
  """

  def feature_extraction_layers(input_):
    """
    Create layers for facial feature extraction
    :param input_: tensor with CN embeddigns
    :return: output layer
    """
    fe1 = tf.layers.dense(input_, 512, activation=tf.nn.sigmoid, name='feature_extraction_layer_1')
    fe2 = tf.nn.l2_normalize(tf.layers.dense(fe1, 256, activation=tf.nn.sigmoid, name='feature_extraction_layer_2'),
                             axis=1, name="normalization_1")
    fe3 = tf.nn.l2_normalize(tf.layers.dense(fe2, 128, activation=tf.nn.tanh, name='feature_extraction_layer_3'),
                             axis=1, name="normalization_2")
    return fe3


  anchors = tf.placeholder(dtype=tf.float32, shape=[None, 2048], name="Anchors")
  positives = tf.placeholder(dtype=tf.float32, shape=[None, 2048], name="Positives")
  negatives = tf.placeholder(dtype=tf.float32, shape=[None, 2048], name="Negatives")
  lr = tf.placeholder(dtype=tf.float32, name="learning_rate")

  anch_final = None
  pos_final = None
  neg_final = None

  with tf.variable_scope('face_feature_extraction') as scope:
    anch_final = feature_extraction_layers(anchors)
    scope.reuse_variables()
    pos_final = feature_extraction_layers(positives)
    neg_final = feature_extraction_layers(negatives)

  # loss = tf.reduce_mean(
  #               tf.square(tf.norm(anch_final - pos_final, axis=1)) + \
  #                 tf.maximum(0., -tf.square(tf.norm(anch_final - neg_final, axis=1)) + alpha)
  #         )

  loss = tf.reduce_mean(  tf.maximum(0., 
                tf.reduce_sum(tf.square(anch_final - pos_final), axis=1, name="positive_penalty") - \
                tf.reduce_sum(tf.square(anch_final - neg_final), axis=1, name="negative_penalty") + tf.constant(alpha, dtype=tf.float32, name="Margin"), name="truncate")
          )

  accuracy = None
  decision_threshold = .8

  with tf.variable_scope('accuracy') as acc_scope:
    pos_sq_norm = tf.reduce_sum(tf.square(anch_final - pos_final), axis=1)
    neg_sq_norm = tf.reduce_sum(tf.square(anch_final - neg_final), axis=1)
    pos_res = tf.less(pos_sq_norm, decision_threshold)
    neg_res = tf.greater(neg_sq_norm, decision_threshold)
    accuracy = tf.reduce_mean(tf.cast(tf.concat([pos_res, neg_res], axis=0), dtype=tf.float32))
    sq_norms_concat = tf.concat([pos_sq_norm, neg_sq_norm], axis=0)
    threshold_check = tf.less(sq_norms_concat, decision_threshold)

  train = tf.train.AdamOptimizer(lr).minimize(loss)

  return {
    'loss': loss,
    'train': train,
    'anchors' : anchors,
    'positives' : positives,
    'negatives' : negatives,
    'accuracy' : accuracy,
    'lr' : lr,
    'thr_ckh' : threshold_check,
    'sq_norms' : sq_norms_concat
  }

  
with tf.Session() as sess:
  # Load ProtoBuf model
  with tf.gfile.GFile(pretrained_graph_path,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

  # Get CNN terminals
  input_ = tf.get_default_graph().get_tensor_by_name("Model_Input:0")
  embedding_ = tf.get_default_graph().get_tensor_by_name("Model_Output:0")
  # embedding_ = tf.get_default_graph().get_tensor_by_name("InceptionResnetV2/Logits/Dropout/Identity:0")

  terminals = {
      'input': input_,
      'output' : embedding_,
      'session' : sess
  }

  trip_loss_term = triplet_loss_model()

  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  writer = tf.summary.FileWriter("./tf_summary", graph=sess.graph)
  # sys.exit()

  # Initialize data reader
  data = LFWDataLoader(data_path, terminals, "test_set.csv")
  # sys.exit()


  for e in range(epochs):
    epoch_finished = False
    test_anc, test_pos, test_neg = data.test_batch

    last_batch_loss = 0.

    learning_rate = 0.0001
    lr_decay = 0.999

    while not epoch_finished:
      (anchors, positive, negative), epoch_finished = data.next_minibatch()
      
      _, last_batch_loss = sess.run([trip_loss_term['train'], trip_loss_term['loss']], {
        trip_loss_term['anchors']: anchors, 
        trip_loss_term['positives']: positive,
        trip_loss_term['negatives']: negative,
        trip_loss_term['lr']: learning_rate
        }
      )

    learning_rate *= lr_decay

    acc, norms, last_batch_loss = sess.run([trip_loss_term['accuracy'],trip_loss_term['sq_norms'],  trip_loss_term['loss']], {
        trip_loss_term['anchors']: test_anc, 
        trip_loss_term['positives']: test_pos,
        trip_loss_term['negatives']: test_neg
        })
    print("Epoch %d, test acc %.4f, test batch loss %.4f" % (e,acc,last_batch_loss))

    

    

  





