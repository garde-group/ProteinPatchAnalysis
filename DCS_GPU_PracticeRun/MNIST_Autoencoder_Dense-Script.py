"""
Created on Thu Jun  6 17:01:04 2019

Python script versions of ipython notebooks 
located in ProteinPatchAnalysis/AutoEncoderNotebooks.
Reads in mnist.pickle that contains mnist dataset
(Cannot import MNIST directly on CCNI). 
Note that the mnist.pickle file is not included in 
github repository because it is too large.
@author: camil
"""

# From Tutorial: https://blog.keras.io/building-autoencoders-in-keras.html

import keras
import pickle

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import mnist
import numpy as np

data = open("mnist.pickle","rb") 
[x_train,x_test] = pickle.load(data)


#
#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
#
#
history = autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']


pickle_out = open("Histories/losses_Dense.pickle","wb")
pickle.dump([loss_values,val_loss_values], pickle_out)
pickle_out.close()


# Save the weights
autoencoder.save_weights('Histories/model_weights.h5')

# Save the model architecture
with open('Histories/model_architecture.json', 'w') as f:
    f.write(autoencoder.to_json())


        
