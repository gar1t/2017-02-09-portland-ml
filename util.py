import tensorflow as tf

def reset_graph():
    tf.reset_default_graph()
    return tf.get_default_graph()

def log_graph(g, path):
    writer = tf.summary.FileWriter(path, graph=g)
    writer.close()

def checkpoint(sess, path):
    saver = tf.train.Saver()
    saver.save(sess, path)

def export_meta(g, path):
    tf.train.export_meta_graph(path + "/export.meta", graph=g)
