import tensorflow as tf

for num in ['0', '1', '2', '3', '4']:
    meta_path = num + '\\model.ckpt-82.meta'
    output_node_names = ['HoleDefect/Classfication/dense/BiasAdd']

    with tf.Session() as sess:
        # Restore the graph
        saver = tf.train.import_meta_graph(meta_path)

        # Load weights
        saver.restore(sess, tf.train.latest_checkpoint(num + '\\'))

        # Freeze the graph
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            output_node_names)

        # Save the frozen graph
        with open('Holedefect' + num + '.pb', 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())
