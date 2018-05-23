import argparse
import tensorflow as tf
import pickle

def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="load/frozen.pb", 
        type=str, help="Frozen model file to import")
    args = parser.parse_args()
    graph = load_graph(args.frozen_model_filename)

    x1 = graph.get_tensor_by_name('prefix/encoder_inputs:0')
    x2 = graph.get_tensor_by_name('prefix/encoder_inputs_length:0')
    x3 = graph.get_tensor_by_name('prefix/batch_size:0')
    y = graph.get_tensor_by_name('prefix/decoder/decoder_pred_eval:0')

    with tf.Session(graph=graph) as sess:
        y_out = sess.run(y, feed_dict={
            x1: [[20, 23, 22, 500, 31, 3000]],
            x2: [5],
            x3: 1
        })

        idx2word = {}
        with open('idx2word.pkl', 'rb') as handle:
            pickle.dump(idx2word, handle)
        print(idx2word)
        predict = [ idx2word[x] for x in y_out ]
        print(predict)
    print ("finish")
