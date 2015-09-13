__author__ = 'Giahy'

from gensim.models import word2vec
from nltk.corpus import brown
from numpy import array
import os

class Word2VecModel:
    def __init__(self, file_name):
        self.embsize = int(file_name.split('_')[-1])
        self.model = None
        if os.path.isfile(file_name):
            self.model = word2vec.Word2Vec.load(file_name)

        if self.model is None:
            model = word2vec.Word2Vec(brown.sents(), size=self.embsize, window=5, min_count=5, workers=4)
            model.save(file_name)

    def getWordEmbeddingFromString(self, word_str):
        try:
            return array(self.model[word_str])
        except:
            return (array(self.model['Join']) + array(self.model['London']))/2

    def parseInstanceFromSentence(self, sentence_str):
        words_str = sentence_str.split(' ')
        return [self.getWordEmbeddingFromString(word_str) for word_str in words_str]


if __name__ == '__main__':
    w2vmodel = Word2VecModel('w2vmodel_100')
    print(w2vmodel.getWordEmbeddingFromString('Join'))

    print(w2vmodel.getWordEmbeddingFromString('London'))

    print((w2vmodel.getWordEmbeddingFromString('Join') + w2vmodel.getWordEmbeddingFromString('London'))/2)



