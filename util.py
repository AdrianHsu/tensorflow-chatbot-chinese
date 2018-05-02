import numpy as np
from colors import *

special_tokens = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}

def inv_sigmoid(num_epo):
    x = np.arange(-1.0, 2.0, (3.0/num_epo))
    y = 1 / (1 + np.e**x)
    print(y)
    return y

def decoder_print(idx2word, _in, _len_in, _out, _len_out, pred, my_color):

    _in = list(reversed(_in))
    eos = len(pred) - 1 
    for i in range(len(pred)):
        if pred[i] == special_tokens['<PAD>']:
            eos = i
            break
    if eos == 0:
        eos = 1 # print out the <EOS>

    prev = [ idx2word[x] for x in _in[0:(_len_in)] ]
    ans  = [ idx2word[x] for x in _out[0:(_len_out)] ]
    pred = [ idx2word[x] for x in pred[0:(eos)] ] 

    print(color('\nQuestions: ' + str(prev) + \
            '\n Answers : ' + str(ans ) + ' (len: ' + str(len(_out)) + ', eos: ' + str(_len_out) + ')' + \
            '\n Predict : ' + str(pred) + ' (len: ' + str(len(pred)) +', eos: ' + str(eos) + ')', fg=my_color))
