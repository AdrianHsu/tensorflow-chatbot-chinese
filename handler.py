import os
import numpy as mp
import pickle
import re
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

data_dir = './data'
filename = '/xaa'

TOTAL_LINE_NUM = 50000
TRAIN_LINE_NUM = 45000
EVAL_LINE_NUUM =  5000

# filename = '/clr_conversation.txt'
# TOTAL_LINE_NUM = 2842478
# TRAIN_LINE_NUM = 2840000
# EVAL_LINE_NUM  =    2478
special_tokens = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
special_tokens_to_word = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']

class DatasetBase:
    def __init__(self):
        self.vocab_num = 0
        self.word2idx = {}
        self.idx2word = {}
        self.data = []

    def sentence_to_idx(self, sent):
        l = []
        unk_num = 0.0
        for word in sent:
            if word in self.word2idx:
                l.append(self.word2idx[word])
            else:
                l.append(special_tokens['<UNK>'])
                unk_num += 1
        if unk_num / float(len(sent)) >= 0.4:
            emp = []
            return emp

        return l

    def prep(self, data):
        init = True
        for i in range(len(data)):
            reg = re.findall(r"[\w']+", data[i])
            if len(reg) == 0: # +++$+++
                init = True
                continue

            sent = text_to_word_sequence(data[i], lower=True, split=' ')
            if len(sent) > 20: # too long
                init = True
                continue
            idx_list = self.sentence_to_idx(sent)
            if len(idx_list) == 0: # <UNK> too many
                init = True
                continue            

            if init:
                _in = idx_list
                init = False
            else:
                _out = idx_list
                _rev_in = list(reversed(_in))
                self.data.append([_rev_in, _out])
                _in = _out

        print('original line num:', len(data))
        print('prep data num: ', len(self.data))


class DatasetTrain(DatasetBase):
    def __init__(self):
        super().__init__()

    def build_dict(self, data_dir, filename, min_count): # for datasetTrain

        file_path = data_dir + filename
        file = open(file_path, 'r')

        raw_line = []
        train_data = []
        eval_data = []
        cnt = 0
        for line in file:
            if cnt < TRAIN_LINE_NUM:
                train_data.append(line)
            else:
                eval_data.append(line)
            cnt += 1
            reg = re.findall(r"[\w']+", line)
            if len(reg) == 0:
                continue
            raw_line.append(line)
        tokenizer = Tokenizer(lower=True, split=' ')
        tokenizer.fit_on_texts(raw_line)

        word_counts = {}
        for tok in tokenizer.word_counts.items():
            if tok[1] >= min_count:
                word_counts[tok[0]] = tok[1]


        for i in range(0, 4):
            tok = special_tokens_to_word[i]
            self.word2idx[tok] = i
            self.idx2word[i] = tok

        cnt = 0
        for tok in tokenizer.word_index.items():
            if tok[0] in word_counts:
                self.word2idx[tok[0]] = cnt + 4
                self.idx2word[cnt + 4] = tok[0]
                cnt += 1

        self.vocab_num = len(self.word2idx)

        with open('word2idx.pkl', 'wb') as handle:
            pickle.dump(self.word2idx, handle)
        with open('idx2word.pkl', 'wb') as handle:
            pickle.dump(self.idx2word, handle)

        return train_data, eval_data
class DatasetEval(DatasetBase):
    def __init__(self):
        super().__init__()
    def load_dict(self): # for datasetEval
        with open('word2idx.pkl', 'rb') as handle:
            self.word2idx = pickle.load(handle)
        with open('idx2word.pkl', 'rb') as handle:
            self.idx2word = pickle.load(handle)

        self.vocab_num = len(self.word2idx)
class DatasetTest(DatasetBase):
    def __init__(self):
        super().__init__()
        test_data = []

    def load_dict(self): # for datasetEval
        with open('word2idx.pkl', 'rb') as handle:
            self.word2idx = pickle.load(handle)
        with open('idx2word.pkl', 'rb') as handle:
            self.idx2word = pickle.load(handle)

        self.vocab_num = len(self.word2idx)
    def load_test_line(self, data_dir, filename):
        file_path = data_dir + filename
        file = open(file_path, 'r')

        test_data = []
        for line in file:
            test_data.append(line)
        return test_data

    def prep(self, data):
        for i in range(len(data)):
            line = data[i]
            reg = re.findall(r"[\w']+", line)
            if len(reg) == 0:
                continue
            sent = text_to_word_sequence(line, lower=True, split=" ")
            _in = self.sentence_to_idx(sent)
            self.test_data.append(list(reversed(_in)))

datasetTrain = DatasetTrain()
train_data, eval_data = datasetTrain.build_dict(data_dir, filename, 5)
datasetTrain.prep(train_data)
datasetEval = DatasetEval()
datasetEval.load_dict()
datasetEval.prep(eval_data)


