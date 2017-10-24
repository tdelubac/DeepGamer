import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

def MLP(input_size, label_size):

    NN = input_data(shape=[None, input_size], name='input')
    
    NN = fully_connected(NN, 256, activation='relu')
    NN = dropout(NN,0.8)

    NN = fully_connected(NN, 128, activation='relu')
    NN = dropout(NN,0.8)

    NN = fully_connected(NN, 32, activation='relu')
    NN = dropout(NN,0.8)

    NN = fully_connected(NN, label_size, activation='softmax')
    NN = regression(NN, optimizer='adam', learning_rate=1e-3, loss='mean_square', name='targets')

    model = tflearn.DNN(NN, tensorboard_dir='log')

    return model

    

