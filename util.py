import numpy as np
from colors import *

special_tokens = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}

def inv_sigmoid(num_epo):
    #x = np.arange(-4.0, 0.0, (4.0/num_epo))
    #y = 1 / (1 + np.e**x)
    y = np.ones(num_epo)
    print(y)
    return y

def decoder_print(idx2word, _in, _len_in, _out, _len_out, pred, my_color):

    _in = list(reversed(_in))
    eos = len(pred) - 1 
    for i in range(len(pred)):
        if pred[i] == special_tokens['<EOS>']:
            eos = i
            break

    prev = [ idx2word[x] for x in _in[0:(_len_in)] ]
    ans  = [ idx2word[x] for x in _out[0:(_len_out)] ]
    predict = [ idx2word[x] for x in pred[0:(len(pred) - 1)] ] 

    print(color('\nQuestions: ' + str(prev) + \
            '\n Answers : ' + str(ans) + ' (len: ' + str(len(_out)) + ', eos: ' + str(_len_out) + ')' + \
            '\n Predict : ' + str(predict) + ' (len: ' + str(len(pred)) +', eos: ' + str(eos) + ')', fg=my_color))
