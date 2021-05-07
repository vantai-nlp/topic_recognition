from preprocessing import load_data, text2words
from feature_extraction import WordEmbedding
from classifier import MultinomialNB_custom, Neural_Network, rnn_text_classification
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
        model.fit(X_train, y_train, epochs = 5, batch_size = 16, validation_data=(X_val, y_val))

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

    

    def _text2vecs(self, model, X):
        list_sen_vec = []
        for i in tqdm(range(len(X))):
            words = text2words([X[i]])[0]
            vec_words = []
            for word in words:
                try:
                    vec_word = model.transform(word)
                    vec_words.append(vec_word)
                except:
                    pass
            #vec_words = np.asarray(vec_words)
            #list_sen_vec.append(vec_words)

            tmp = 30 - len(vec_words)
            if tmp > 0:
                for i in range(tmp):
                    vec_word = np.zeros(shape = 300)
                    vec_words.append(vec_word)
            elif tmp < 0:
                vec_words = vec_words[:30]
            list_sen_vec.append(vec_words)

        list_sen_vec = np.asarray(list_sen_vec)
        return list_sen_vec

    def _pipeline_w2v_and_recurentneuralnetwork(self, dir_dataset):

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


        # transform text to vector by word to vec pretrain model
        W2V = WordEmbedding().Word2Vec()
        X_train = self._text2vecs(W2V, X_train)
        X_val = self._text2vecs(W2V, X_val)
        X_test = self._text2vecs(W2V, X_test)
        

        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)

        input_dim, classes = len(X_train[0][0]), len(categories) 
        model =  rnn_text_classification(input_dim, classes).model

        model.fit(X_train, y_train, epochs = 2, batch_size = 16, validation_data=(X_val, y_val))


        y_hat = model.predict(X_test)
        y_hat = np.argmax(y_hat, axis = 1)

        print(classification_report(y_test, y_hat))
        return categories, W2V, model    


    def pipeline_w2v_and_recurentneuralnetwork(self, dir_dataset):
        categories, W2V, model = self._pipeline_w2v_and_recurentneuralnetwork(dir_dataset)
        command = input('Sentence: ')
        while command != 'exit':
            x = self._text2vecs(W2V, [command])
            y_hat = model.predict(x)
            y_hat = np.argmax(y_hat)
            print('-> Category is: {}\n'.format(categories[y_hat]))
            command = input('Sentence: ')
            
    





