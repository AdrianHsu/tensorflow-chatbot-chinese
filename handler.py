import os
import numpy as np
import pickle
import re
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

np.random.seed(0)

special_tokens = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
special_tokens_to_word = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']

class Batch:
    def __init__(self, batch_size = 0):
        self.batch_size = batch_size
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []
    
    def printBatch(self): # debug
        for i in range(self.batch_size):
            print(self.encoder_inputs[i])
            print(self.encoder_inputs_length[i])
            print(self.decoder_targets[i])
            print(self.decoder_targets_length[i])
            print("-----")

class DatasetBase:
    def __init__(self):
        self.vocab_num = 0
        self.word2idx = {}
        self.idx2word = {}
        self.data = []
        self.perm = []
        self.ptr = 0

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
                assert len(_rev_in) != 0
                assert len(_out) != 0
                self.data.append([_rev_in, _out])
                _in = _out
            if i % 100000 == 0:
                print("building data list: " + str(i) + "/" + str(len(data)) + " done.")


        print('original line num:', len(data))
        print('prep data num: ', len(self.data))
        self.data = np.array(self.data)
        self.perm = np.arange( len(self.data), dtype=np.int )
        self.shuffle_perm()

    def shuffle_perm(self):
        np.random.shuffle(self.perm)

    def next_batch(self, batch_size, shuffle = True):

        ptr = self.ptr
        max_size = len(self.data)
        if ptr + batch_size <= max_size:
            if shuffle:
                d_list = self.data[self.perm[ptr:(ptr + batch_size)]]
            else:
                d_list = self.data[ptr:(ptr + batch_size)]
            
            self.ptr += batch_size
        else:
            right = batch_size - (max_size - ptr)
            if shuffle:
                d_list = np.concatenate((self.data[self.perm[ptr:max_size]] , self.data[self.perm[0:right]]), axis=0)
            else:
                d_list = np.concatenate((self.data[ptr:max_size] , self.data[0:right]), axis=0)
            self.ptr = right

        return self.create_batch(d_list, batch_size)

    def create_batch(self, samples, batch_size):
        batch = Batch(batch_size)
        batch.encoder_inputs_length = [len(sample[0]) for sample in samples]
        batch.decoder_targets_length = [len(sample[1]) for sample in samples]
        max_source_length = max(batch.encoder_inputs_length)
        max_target_length = max(batch.decoder_targets_length)
        for sample in samples:
            source = sample[0]
            pad = [special_tokens['<PAD>']] * (max_source_length - len(source))
            batch.encoder_inputs.append(pad + source)

            target = sample[1]
            if len(target) < max_target_length:
                eos = [special_tokens['<EOS>']] * 1
                pad = [special_tokens['<PAD>']] * (max_target_length - len(target) - 1)
                batch.decoder_targets.append(target + eos + pad)
            else:
                pad = []
                batch.decoder_targets.append(target + pad)

        return batch

    def load_dict(self): # for datasetEval
        with open('word2idx.pkl', 'rb') as handle:
            self.word2idx = pickle.load(handle)
        with open('idx2word.pkl', 'rb') as handle:
            self.idx2word = pickle.load(handle)

        self.vocab_num = len(self.word2idx)

class DatasetTrain(DatasetBase):
    def __init__(self):
        super().__init__()
        
    def build_dict(self, data_dir, filename, min_count,
                train_line_num, eval_line_num, PKL_EXIST=False):
        # for datasetTrain

        file_path = data_dir + filename
        file = open(file_path, 'r')

        raw_line = []
        train_data = []
        eval_data = []
        cnt = 0
        for line in file:
            if cnt < train_line_num:
                train_data.append(line)
            else: 
                eval_data.append(line)
            cnt += 1
            reg = re.findall(r"[\w']+", line)
            if len(reg) == 0:
                continue
            raw_line.append(line)
        
        assert len(train_data) == train_line_num
        assert len(eval_data)  == eval_line_num
        if PKL_EXIST:
            print('dict already exists, loading...')
            self.load_dict()
            return train_data, eval_data

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

class DatasetTest(DatasetBase):
    def __init__(self):
        super().__init__()
        test_data = []

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

        print('test data num: ', len(self.test_data))


