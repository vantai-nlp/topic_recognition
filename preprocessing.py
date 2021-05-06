# load data from file txt, input: file name, output: X, y
def load_data(dir_file, max_size = 2000):
        category = dir_file.split('/')[-2].split('.')[0]
        file = open(dir_file, 'r')
        X = file.readlines()

        if len(X) > max_size:
            X = X[:max_size]

        y = [category for i in range(len(X))]
        return X, y

# transform text to list of words, input: texts, output: n rows - each row is a vector of word 
def text2words(texts):
    matrix = []
    for text in texts:
        words = text.split()
        matrix.append(words)
    return matrix
    