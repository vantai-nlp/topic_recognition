from model import model

if __name__ == '__main__':
    dir_dataset = 'dataset/English'
    model().pipeline_bow_and_multinomialNB(dir_dataset)
    #model().pipeline_bow_and_neuralnetwork(dir_dataset)