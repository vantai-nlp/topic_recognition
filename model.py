from preprocessing import load_data, text2words
from feature_extraction import WordEmbedding
from classifier import MultinomialNB_custom, Neural_Network
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical
import os
from tqdm import tqdm
from keras.models import load_model

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
        while command != 'q':
            x = BoW.transform([command])
            y_hat = model.predict(x)
            print('-> Category is: {}\n'.format(y_hat[0]))
            command = input('Sentence: ')
            





