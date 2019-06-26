# -*- coding: utf-8 -*-
"""
Created on Tue Jun  18 2019

Main code

3D function conv variational autoencoder for protein patches
taking in hyperparameters


@author: camil
"""

import keras
import pickle 
import numpy as np
import tensorflow as tf

from keras import layers
from keras.layers import Input, Dense, Conv3D, MaxPooling3D, UpSampling3D, Lambda
from keras.models import Model
from keras import optimizers
from keras import regularizers
from keras import backend as K
import time

from gen_batches import gen_batches, gen_batches_validation

def Conv_VAE3D(n_epochs=2,
             batch_size=10,
             learning_rate=0.001,
             decay_rate=0.0,
             latent_dim=8,
             name='stats.pickle'):

    # Prepare session:
    K.clear_session()

    # Number of samples to use for training and validation:
    n_train = 1500
    n_val = 1000

# ENCODER: ---------------------------------------------------------------

    input_img = Input(shape=(50,50,50,4),name="Init_Input") 
    x = layers.Conv3D(32, (3, 3, 3) , padding="same", activation='relu',name='E_Conv1')(input_img)
    x = layers.MaxPooling3D((2,2,2),name='E_MP1')(x)
    x = layers.Conv3D(64, (3, 3, 3), padding="same", activation='relu',name='E_Conv2')(x)
    x = layers.MaxPooling3D((2,2,2),name='E_MP2')(x)
    x = layers.Conv3D(64, (3, 3, 3), padding="valid", activation='relu',name='E_Conv3')(x)
    x = layers.MaxPooling3D((2,2,2),name='E_MP3')(x)
    x = layers.Conv3D(128, (3, 3, 3), padding="same", activation='relu',name='E_Conv4')(x)
    
    shape_before_flattening = K.int_shape(x)
        
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)
    
    encoder = Model(input_img, x)  

    print(encoder.summary())
    
# VARIATIONAL LAYER: ------------------------------------------------------
    
    z_mean = layers.Dense(latent_dim,name='V_Mean')(x)
    z_log_var = layers.Dense(latent_dim,name='V_Sig')(x)
    
    
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),mean=0., stddev=1.)
        return z_mean + K.exp(z_log_var) * epsilon
    
    z = layers.Lambda(sampling,name='V_Var')([z_mean, z_log_var])
    variation = Model(input_img,z)
    
    
    print(variation.summary())
# DECODER: ---------------------------------------------------------------

    decoder_input = layers.Input(shape=(latent_dim,),name='D_Input')
    
    x = layers.Dense(np.prod(shape_before_flattening[1:]),activation='relu',name='D_Dense')(decoder_input)
    x = layers.Reshape(shape_before_flattening[1:],name='D_UnFlatten')(x)
    x = layers.Conv3DTranspose(32, 3, padding='same',activation='relu',name='D_DeConv1')(x)
    x = layers.UpSampling3D((2,2,2))(x)
    x = layers.Conv3D(4, 3,padding='same',activation='sigmoid',name='D_Conv1')(x)
    x = layers.UpSampling3D((5,5,5))(x)
    x = layers.Conv3D(4, 3,padding='same',activation='sigmoid',name='D_Conv2')(x)

    decoder = Model(decoder_input, x)
    
    print(decoder.summary())
        
# CALLBACKS: --------------------------------------------------------------

    class TimeHistory(keras.callbacks.Callback):
        start =[]
        end=[]
        times=[]
    
        def on_epoch_begin(self, batch, logs=None):
            self.start=time.time()
    
        def on_epoch_end(self, batch, logs=None):
            self.end=time.time()
            self.times.append(self.end-self.start)
    
    
# CUSTOM LAYERS: ----------------------------------------------------------
    
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

# DEFINE FINAL MODEL: ----------------------------------------------------

    z_encoded = variation(input_img)
    z_decoded = decoder(z_encoded)

    # Construct Final Model:
    y = CustomVariationalLayer()([input_img, z_decoded])
    vae = Model(input_img, y)
    
    print(vae.summary())


    # Define Optimizer:    
    vae_optimizer=keras.optimizers.Adam(lr=learning_rate, 
                                        beta_1=0.9, 
                                        beta_2=0.999, 
                                        decay=decay_rate, 
                                        amsgrad=False)
    
    vae.compile(optimizer=vae_optimizer, loss=None) # Not using custom vae loss function defined above
    
    # Define time callback:
    time_callback = TimeHistory()    
    
    steps = n_train // batch_size
    val_steps = n_val // batch_size
# FIT MODEL: --------------------------------------------------------------
    history = vae.fit_generator(gen_batches(batch_size),
                                shuffle=True,
                                epochs=n_epochs,
                                steps_per_epoch= steps,
                                callbacks = [time_callback],
                                validation_data = gen_batches_validation(batch_size),
                                validation_steps = val_steps
                                )    
    
# OUTPUTS: -------------------------------------------------------------
    
    history_dict = history.history
    
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    times = time_callback.times
    data = {'train_loss': loss_values, 'val_loss':val_loss_values,'epoch_time':times}

    pickle_out = open(name,"wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()
  
    K.clear_session()  
    return(history_dict)
