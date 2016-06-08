# coding=utf-8
import common as cm
from jieba import analyse
import jieba as jb
import datetime
import sys
import os

reload(sys)
sys.setdefaultencoding("utf8")
import pandas as pd  # 导入Pandas
import numpy as np  # 导入Numpy
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.models import model_from_json


os.chdir('/Users/romber/Documents')
user = 'bigdata'
now = datetime.datetime.now()
tain_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_name = 'sa_model_' + tain_time
max_len = 50


class Analyze:
    def __init__(self, type):
        self.t = type
        self.result = cm.get_comments(self.t)
        self.y = []
        self.x = []
        self.vec = []
        self.dict = []

    def split_comments(self):
        dict = {}
        for row in self.result:
            content = row[1].encode('UTF-8')
            content = content.replace('\xc2\xa0', '')
            content = list(jb.cut(content))
            self.y.append(int(row[0]))
            self.x.append(content)
            for word in content:
                dict[word] = dict.get(word, 0) + 1

        return dict

    def word2vec(self):
        dict = self.split_comments()

        if self.t == 'train':
            list_sorted = sorted(dict.iteritems(), key=lambda d: d[1], reverse=True)
            cm.update_words(list_sorted)
        elif self.t == 'predict':
            list_sorted = cm.get_words()

        # print '=======print list_sorted======'
        # for word in list_sorted:
        #     print word[0], word[1],
        # print

        for i in range(len(self.x)):
            for j in range(len(self.x[i])):
                seq = 0
                for k in range(len(list_sorted)):
                    if self.x[i][j] == list_sorted[k][0]:
                        seq = k + 1
                self.x[i][j] = seq

        return list_sorted

    def train_model(self):
        print '=======begin to prepare data at ' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '========='
        list_sorted = self.word2vec()

        self.y = np.array(list(self.y))
        self.x = list(sequence.pad_sequences(list(self.x), maxlen=max_len))
        self.x = np.array(list(self.x))
        print '=======end to prepare data at ' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '========='

        print '=======begin to train model at ' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '========='
        model = Sequential()
        model.add(Embedding(input_dim=len(list_sorted) + 1, output_dim=256, input_length=max_len))
        model.add(LSTM(128))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        model.fit(self.x, self.y, batch_size=16, nb_epoch=10)

        json_string = model.to_json()
        open('sa_model_architecture.json', 'w').write(json_string)
        model.save_weights('sa_model_weights.h5', overwrite=True)
        print '=======end to train model at ' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '========='

        return model

    def predict_emotion(self):
        print '=======begin to predict emotion at ' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '========='
        self.word2vec()
        self.x = list(sequence.pad_sequences(list(self.x), maxlen=max_len))
        self.x = np.array(list(self.x))
        model = model_from_json(open('sa_model_architecture.json').read())
        model.load_weights('sa_model_weights.h5')
        model.compile(loss='binary_crossentropy', optimizer='adam')
        p_label = model.predict_classes(self.x)
        p_label = p_label.tolist()
        for i in range(len(p_label)):
            p_label[i] = p_label[i][0]

        cm.update_predict(p_label, self.y)
        print '=======end to predict emotion at ' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '========='

        return p_label


if __name__ == '__main__':
    train = Analyze('train')
    predict = Analyze('predict')

#train.train_model()
predict.predict_emotion()
# acc = np_utils.accuracy(p_label, predict.y)
