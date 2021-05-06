from sklearn.feature_extraction.text import CountVectorizer
import gensim


# using pretrain model to embeded a word to vector
class Word2Vec_custom(object):
    def __init__(self, dir_pretrain_model = './GoogleNews-vectors-negative300.bin'):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(dir_pretrain_model, binary=True)

    def transform(self, word):
        return self.model[word]

    def similarity(self, word_one, word_two):
        return self.model.similarity(word_one, word_two)

# inherit from countvector of keras
class CountVectorizer_custom(CountVectorizer):
    pass

# class contain class countvectorizer and class word2vec
class WordEmbedding(object):
    def CountVectorizer(self):
        model = CountVectorizer_custom()
        return model

    def Word2Vec(self):
        model = Word2Vec_custom()
        return model


