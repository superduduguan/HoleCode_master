import tensorflow as tf


meta_path = 'model.ckpt-82.meta'
output_node_names = ['HoleDetection/LocationResult']   

with tf.Session() as sess:
    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess, tf.train.latest_checkpoint('C:\\Users\\pc\\Desktop\\HoleCode_master\\HolePosition\\'))

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('3and2x2.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())
