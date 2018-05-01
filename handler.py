import os
import numpy as mp
import pickle
import re
from keras.preprocessing.text import Tokenizer

data_dir = './data'
filename = '/xaa'

TOTAL_LINE_NUM = 50000
TRAIN_LINE_NUM = 45000
EVAL_LINE_NUUM =  5000

# filename = '/clr_conversation.txt'
# TOTAL_LINE_NUM = 2842478
# TRAIN_LINE_NUM = 2840000
# EVAL_LINE_NUM  =    2478

special_tokens_to_word = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']

class Dataset:
    def __init__(self):
        self.vocab_num = 0
        self.word2idx = {}
        self.idx2word = {}


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

    def load_dict(self): # for datasetEval
        with open('word2idx.pkl', 'rb') as handle:
            self.word2idx = pickle.load(handle)
        with open('idx2word.pkl', 'rb') as handle:
            self.idx2word = pickle.load(handle)

        self.vocab_num = len(self.word2idx)

dataset = Dataset()
train_data, eval_data = dataset.build_dict(data_dir, filename, 5)

print(len(train_data))
print(len(eval_data))


