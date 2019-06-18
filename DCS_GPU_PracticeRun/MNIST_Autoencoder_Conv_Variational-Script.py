# -*- coding: utf-8 -*-
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

import keras
import pickle 
import numpy as np

from keras import layers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda
from keras.models import Model
from keras import regularizers
from keras import backend as K

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

data = open("mnist.pickle","rb")
[x_train,x_test] = pickle.load(data)



x_train = x_train.reshape((60000, 28, 28) + (1,))
x_test = x_test.reshape((10000, 28, 28) + (1,))

batch_size = 128
latent_dim = 2


x = layers.Conv2D(32, 3,padding='same', activation='relu')(input_img)
x = layers.Conv2D(64, 3,padding='same', activation='relu',strides=(2, 2))(x)
x = layers.Conv2D(64, 3,padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3,padding='same', activation='relu')(x)
shape_before_flattening = K.int_shape(x)
x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)

encoder = Model(input_img, x)
x_encoded = encoder(input_img)

z_mean = layers.Dense(latent_dim)(x_encoded)
z_log_var = layers.Dense(latent_dim)(x)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

decoder_input = layers.Input(K.int_shape(z)[1:])
x = layers.Dense(np.prod(shape_before_flattening[1:]),activation='relu')(decoder_input)
x = layers.Reshape(shape_before_flattening[1:])(x)
x = layers.Conv2DTranspose(32, 3,padding='same',activation='relu',strides=(2, 2))(x)
x = layers.Conv2D(1, 3,padding='same',activation='sigmoid')(x)

decoder = Model(decoder_input, x)
z_decoded = decoder(z)


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class CustomVariationalLayer(keras.layers.Layer):
    
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss+kl_loss) #xent_loss) # + kl_loss)
    
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x
    
y = CustomVariationalLayer()([input_img, z_decoded])

vae = Model(input_img, y)
vae.compile(optimizer='rmsprop', loss=None)


time_callback = TimeHistory()

history = vae.fit(x=x_train, y=None,shuffle=True,epochs=50,batch_size=batch_size,validation_data=(x_test, None),callbacks=[time_callback])


history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']


pickle_out = open("Histories/losses-xxx.pickle","wb")
pickle.dump([loss_values,val_loss_values], pickle_out)
pickle_out.close()

times = time_callback.times
pickle_out = open("Histories/times-xxx.pickle","wb")
pickle.dump([loss_values,val_loss_values], pickle_out)
pickle_out.close()




# Save the weights
vae.save_weights('Histories/model_weights-xxx.h5')

# Save the model architecture
with open('Histories/model_architecture.json', 'w') as f:
    f.write(vae.to_json())




