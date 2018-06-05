import tensorflow as tf
import jieba

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
    print('load graph: done')
    return graph

def seq(test_input, word2idx):
    test_input = jieba.cut(test_input, cut_all=False)
    test_input = ",".join(test_input)
    test_input = test_input.split(',')
    if test_input[-1] != "。":
        test_input.append("。")
    # print(test_input)
    input_id = []
    for x in test_input:
        if x in word2idx:
            input_id.append(word2idx[x])
        else:
            input_id.append(3) # <UNK>

    return input_id

def dec(predict, idx2word):
    predict = [ idx2word[x] for x in predict ]
    if predict[-1] == "<EOS>":
        predict = predict[:-1]
    sen = []
    for word in predict:
        if len(sen) == 0:
            sen.append(word)
            continue
        if word == sen[-1]:
            continue
        if word == '<UNK>':
            continue
        sen.append(word)
    return "".join(sen)

# if __name__ == '__main__':
#     graph = load_graph("frozen.pb",)

    # x1 = graph.get_tensor_by_name('prefix/encoder_inputs:0')
    # x2 = graph.get_tensor_by_name('prefix/encoder_inputs_length:0')
    # x3 = graph.get_tensor_by_name('prefix/batch_size:0')
    # y = graph.get_tensor_by_name('prefix/decoder/decoder_pred_eval:0')
    
    # word2idx = {}
    # with open('word2idx.pkl', 'rb') as handle:
    #     word2idx = pickle.load(handle)

    
    # idx2word = {}
    # with open('idx2word.pkl', 'rb') as handle:
    #     idx2word = pickle.load(handle)
    
#     with tf.Session(graph=graph) as sess:
#         while True:
#             test_input = input("Q: ")
#             input_id = seq(test_input, word2idx)
#             y_out = sess.run(y, feed_dict={
#                 x1: [input_id],
#                 x2: [len(input_id)],
#                 x3: 1
#             })

#             sen = dec(y_out[0], idx2word)

#             print("A: " + sen)
    
#     print ("finish")
