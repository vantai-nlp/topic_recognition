from preprocessing import load_data, text2words
from feature_extraction import WordEmbedding
from classifier import MultinomialNB_custom, Neural_Network
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical
import os
from tqdm import tqdm
from keras.models import load_model
import numpy as np

class model(object):
    def __init__(self):
        pass

    def load_model(self, dir_pretrain_model):
        return load_model(dir_pretrain_model)

    def _pipeline_bow_and_multinomialNB(self, dir_dataset):

        categories = os.listdir(dir_dataset)

        # X_train
        X_train, y_train = [], []
        for category in tqdm(categories):
            path = os.path.join(dir_dataset, category, 'evc.train.en')
            X_tmp, y_tmp = load_data(path)
            for i in range(len(X_tmp)):
                X_train.append(X_tmp[i])
                y_train.append(y_tmp[i])

        
        # X_test
        X_test, y_test = [], []
        for category in tqdm(categories):
            path = os.path.join(dir_dataset, category, 'evc.test.en')
            X_tmp, y_tmp = load_data(path)
            for i in range(len(X_tmp)):
                X_test.append(X_tmp[i])
                y_test.append(y_tmp[i])


        # transform text to vector by word of bag
        BoW = WordEmbedding().CountVectorizer()
        X_train = BoW.fit_transform(X_train)
        X_train = X_train.toarray()
        print(X_train.shape, len(y_train))

        X_test = BoW.transform(X_test).toarray()

        model = MultinomialNB_custom()
        model.fit(X_train, y_train)

        y_hat = model.predict(X_test)
        print(classification_report(y_test, y_hat))
        return BoW, model

    def pipeline_bow_and_multinomialNB(self, dir_dataset):
        BoW, model = self._pipeline_bow_and_multinomialNB(dir_dataset)
        command = input('Sentence: ')
        while command != 'exit':
            x = BoW.transform([command])
            y_hat = model.predict(x)
            print('-> Category is: {}\n'.format(y_hat[0]))
            command = input('Sentence: ')


    
    def _pipeline_bow_and_neuralnetwork(self, dir_dataset):

        categories = os.listdir(dir_dataset)

        # X_train
        X_train, y_train = [], []
        for category in tqdm(categories):
            path = os.path.join(dir_dataset, category, 'evc.train.en')
            X_tmp, y_tmp = load_data(path)
            for i in range(len(X_tmp)):
                X_train.append(X_tmp[i])
                y_train.append(categories.index(category))

        # X_val
        X_val, y_val = [], []
        for category in tqdm(categories):
            path = os.path.join(dir_dataset, category, 'evc.dev.en')
            X_tmp, y_tmp = load_data(path)
            for i in range(len(X_tmp)):
                X_val.append(X_tmp[i])
                y_val.append(categories.index(category))

        
        # X_test
        X_test, y_test = [], []
        for category in tqdm(categories):
            path = os.path.join(dir_dataset, category, 'evc.test.en')
            X_tmp, y_tmp = load_data(path)
            for i in range(len(X_tmp)):
                X_test.append(X_tmp[i])
                y_test.append(categories.index(category))


        # transform text to vector by word of bag
        BoW = WordEmbedding().CountVectorizer()
        X_train = BoW.fit_transform(X_train)
        X_train = X_train.toarray()
        X_val = BoW.transform(X_val).toarray()
        X_test = BoW.transform(X_test).toarray()
        

        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)

        
        model = Neural_Network(len(X_train[0]), len(categories)).model
        model.summary()
        model.fit(X_train, y_train, epochs = 2, batch_size = 16, validation_data=(X_val, y_val))

        y_hat = model.predict(X_test)
        y_hat = np.argmax(y_hat, axis = 1)

        print(classification_report(y_test, y_hat))
        return categories, BoW, model


    def pipeline_bow_and_neuralnetwork(self, dir_dataset):
        categories, BoW, model = self._pipeline_bow_and_neuralnetwork(dir_dataset)
        command = input('Sentence: ')
        while command != 'exit':
            x = BoW.transform([command])
            y_hat = model.predict(x)
            y_hat = np.argmax(y_hat)
            print('-> Category is: {}\n'.format(categories[y_hat]))
            command = input('Sentence: ')

    

            
    





