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

    test_input = ['只' ,'需' ,'小小的', '推動', ',', '人', '就', '飄向', '那裏' ,'了']

    word2idx = {}
    with open('word2idx.pkl', 'rb') as handle:
        word2idx = pickle.load(handle)
    print(test_input)
    #print(word2idx['只'])
    input_id = []
    for x in test_input:
        if x in word2idx:
            input_id.append(word2idx[x])
        else:
            input_id.append(3) # <UNK>

    print(input_id)
    with tf.Session(graph=graph) as sess:
        y_out = sess.run(y, feed_dict={
            x1: [input_id],
            x2: [len(input_id)],
            x3: 1
        })

        idx2word = {}
        with open('idx2word.pkl', 'rb') as handle:
            idx2word = pickle.load(handle)
        print(y_out)
        predict = [ idx2word[x] for x in y_out[0] ]
        print(predict)
    print ("finish")
