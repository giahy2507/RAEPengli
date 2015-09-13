
__author__ = 'Giahy'

from gensim.models import word2vec
from nltk.corpus import brown
from numpy import array
import os

class Instance:
    def __init__(self,words,embsize):
        self.words = words
        self.embsize = embsize

class Word2VecModel:
    def __init__(self, file_name):
        self.embsize = int(file_name.split('_')[-1])
        self.model = None
        if os.path.isfile(file_name):
            word2vec.FAST_VERSION = 1
            self.model = word2vec.Word2Vec.load(file_name)

        if self.model is None:

            model = word2vec.Word2Vec(brown.sents(), size=self.embsize)
            model.save(file_name)

    def getWordEmbeddingFromString(self, word_str):
        try:
            return array(self.model(word_str))
        except:
            return None

    def parseInstanceFromSentence(self, sentence_str):
        words_str = sentence_str.split

if __name__ == '__main__':
    w2vmodel = Word2VecModel('w2vmodel_100')
    print(w2vmodel.getEmbeddingFromString('American'))



