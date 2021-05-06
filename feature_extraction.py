from sklearn.feature_extraction.text import CountVectorizer
import gensim



class Word2Vec_custom(object):
    def __init__(self, dir_pretrain_model = './GoogleNews-vectors-negative300.bin'):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(dir_pretrain_model, binary=True)

    def transform(self, word):
        return self.model[word]

    def similarity(self, word_one, word_two):
        return self.model.similarity(word_one, word_two)


class CountVectorizer_custom(CountVectorizer):
    pass

class WordEmbedding(object):
    def CountVectorizer(self):
        model = CountVectorizer_custom()
        return model

    def Word2Vec(self):
        model = Word2Vec_custom()
        return model



"""
a = WordEmbedding().CountVectorizer()
d = a.fit_transform(['today hi', 'my name is Van Tai'])
d = d.toarray()
print(d)
print(d.shape)


a = WordEmbedding().Word2Vec()
d = a.transform(['my', 'name', 'is'])
print(d.shape)

"""
