import os

# load data from file txt
def load_data(name_file, max_size = 10):
        category = name_file.split('/')[-2].split('.')[0]
        file = open(name_file, 'r')
        X = file.readlines()

        if len(X) > max_size:
            X = X[:max_size]

        y = [category for i in range(len(category))]
        return X, y

# transform text to list of words
def text2words(text):
    words = text.split()
    return words

"""
X, y = load_data('./dataset/English/Computer/evc.train.en')
text = 'hi everyone , my name is Van Tai .\n'
print(text2words(text))"""
