import tensorflow as tf

def export_protobuf_graph(protobuf_model_path, export_location):
    """
    Loads protobuf model and exports graph ready for tensorboard
    :param protobuf_model_path: Location of ProtoBuf model file. Example: InseptionResnetV2.pb
    :param export_location: Location to store tensorflow graph. Example: ./tf_summary
    :return: None
    """
    with tf.Session() as persisted_sess:
        print("load graph")
        with tf.gfile.GFile(protobuf_model_path,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            persisted_sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            writer = tf.summary.FileWriter(export_location, graph=persisted_sess.graph)



#######################
# Loading and freezing
#######################

def protobuf_from_checkpoint(model_name, checkpoint_location, export_location):
    """
    Assembles the model and saves pretrained model to ProtoBuf format. The last layers are discarded.
    The model and weights are provided by https://github.com/tensorflow/models/tree/master/research/slim
    :param model_name: Determines which architecture to use. Options: ["InceptionV3", "InceptionResnetV2"]
    :param checkpoint_location: location of pretrained weights for the current model
    :param export_location: Location to store ProtoBuf model
    :return: None
    """

    im_input = tf.placeholder(shape=(None, 299, 299, 3), dtype=tf.float32, name="Model_Input")

    if model_name == "InceptionV3":
        from architectures.inception_v3 import inception_v3, inception_v3_arg_scope
        model = inception_v3
        model_scope = inception_v3_arg_scope
        output_node_names = ["Model_Input", "Model_Output"]


    elif model_name == "InceptionResnetV2":
        from architectures.inseption_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
        model = inception_resnet_v2
        model_scope = inception_resnet_v2_arg_scope
        output_node_names = ["Model_Input", "InceptionResnetV2/Logits/Dropout/Identity"]


    else:
        raise NotImplemented("The only supported models are [\"InceptionV3\", \"InceptionResnetV2\"]")



    slim = tf.contrib.slim

    with tf.Session() as sess:
        with slim.arg_scope(model_scope()):
            if model_name == "InceptionV3":
                logits, terminals = model(im_input, is_training=False, create_aux_logits=True, num_classes=1001)
                output = tf.squeeze(terminals['AvgPool_1a'], axis=[1, 2], name='Model_Output')
            elif model_name == "InceptionResnetV2":
                logits, terminals = model(im_input, is_training=False, create_aux_logits=True)

            saver = tf.train.Saver()

            # Restore variables from disk.
            saver.restore(sess, checkpoint_location)
            sess.graph.as_default()
            print("Model restored.")

            # Write graph to tensorboard
            writer = tf.summary.FileWriter("./tf_summary", graph=sess.graph)

            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,  # The session is used to retrieve the weights
                tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
                output_node_names  # The output node names are used to select the usefull nodes
            )

            with tf.gfile.GFile(export_location, "wb") as f:
                f.write(output_graph_def.SerializeToString())


# inception_resnet_v2_model_ckpt = "/Volumes/External/data_sink/inception_resnet_v2_2016_08_30.ckpt"
# inceptionv3_model_ckpt = "/Volumes/External/data_sink/inception_v3.ckpt"







