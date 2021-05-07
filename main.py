from model import model
from feature_extraction import WordEmbedding
from preprocessing import text2words
from classifier import rnn_text_classification
import numpy as np

if __name__ == '__main__':
    dir_dataset = 'dataset/English'
    #model().pipeline_bow_and_multinomialNB(dir_dataset)
    #model().pipeline_bow_and_neuralnetwork(dir_dataset)
    model().pipeline_w2v_and_recurentneuralnetwork(dir_dataset)