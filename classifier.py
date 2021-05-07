from sklearn.naive_bayes import MultinomialNB
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# inherit from class MultinomialNB from kerass
class MultinomialNB_custom(MultinomialNB):
    pass

# build a custom neural network from keras
class Neural_Network(object):
    def __init__(self, input_dim, classes):
        self.model = Sequential()
        self.model.add(Dense(128,input_dim = input_dim, activation = 'relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(classes, activation = 'softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()


# build a custom recurrent neural network from keras
class rnn_text_classification(object):
    def __init__(self, input_dim, classes):
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape = (None, input_dim)))
        self.model.add(Dense(classes, activation = 'softmax'))
        self.model.build()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()