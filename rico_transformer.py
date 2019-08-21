import numpy as np
import h5py
import global_var
import random
import matplotlib.pyplot as plt

import keras
from keras import backend as K
from keras.layers import Dense, Dropout, Flatten, Activation, Input, Add, Dot
from keras.models import Sequential
from keras import losses
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model, Model

#Creation of a single Attention layer
#it will be used in the (input/predicted_output)
#Self-Attention and in the Encoder-Decoder Attention
def Attention(query, keys, value):
	#Simple dot product between query and keys, normalized by
	#their size (50)
    x = Dot(axes = 0)([query, keys])/np.sqrt(50)
    
    #This layer tries to find which part of past is important
    x = Activation('softmax')(x)
	
    output = Dot(axes = [0,1])([x, value])

    return output

#A simple Feed-Foward Neural Network with 3 hidden layers
def NN(input,output_size):
    x = Dense(100, activation = 'relu', use_bias = True)(input)
    x = Dense(80, activation = 'relu', use_bias = True)(x)
    x = Dense(50, activation = 'relu', use_bias = True)(x)
    x = Dense(output_size, activation = 'linear', use_bias = True)(x)
    return x

#Enconding the input information
def encoder(input):
	#Magical creatures that we need to use in the attention layer
    query = Dense(50)(input)
    value = Dense(50)(input)
    keys = Dense(50)(input)
	
	#The attention layer should learn which patterns from the past
	#she needs to learn
    x = Attention(query, keys, value)
    x = Add()([x,input])
    x = BatchNormalization()(x)
	
	#Using the Neural Network
    x = NN(x,50)
    x = Add()([x,input])
    x = BatchNormalization()(x)

    return x

#Decoding the encoded information mixed with the predicted output
def decoder(input, encoded):

    query = Dense(4)(input)
    value = Dense(4)(input)
    keys = Dense(4)(input)
	
	#Encoded information used as keys and values
	#in the Encoder-Decoder Attention layer
    keys_enc = Dense(4)(encoded)
    values_enc = Dense(4)(encoded)
	
	#PROBLEM: for some reason the dot product in the Attention 
	#layer is having a rank incompability (1,4) - (?,4)
    print(query.shape == keys_enc.shape)
    x = Attention(query, keys, value)
    x = Add()([x,input])
    x = BatchNormalization()(x)
	
    x = NN(x,4)
    x = Add()([x,input])
    x = BatchNormalization()(x)
	
	#Ouput return
    x = Dense(4, activation='linear')(x)

    return x

#Loading the data
"""
N = global_var.N
M = global_var.M

data = h5py.File('stochastic_heston.h5','r')
data.require_dataset("Heston",((20*N**4),1001,2),dtype='float32')
#shuffled_data = data["Heston"][...]
#np.random.shuffle(shuffled_data)


x1, y1 = data["Heston"][:int(0.85*M),900:950,:], data["Heston"][:int(0.85*M),950:954,0]
x2, y2 = data["Heston"][int(0.85*M):,900:950,:], data["Heston"][int(0.85*M):,950:954,0]
# TODO: redo the input- output scheme

x_test = np.zeros((4*5000,50))
x_train = np.zeros((4*10000,50))
y_test = np.zeros((4*5000,4))
y_train = np.zeros((4*10000,4))

for i in range(10000):
    for j in range(4):
        x_train[4*i+j,:] = x2[i,:,0]
        y_train[4*i+j,:j+1] = y2[i,:j+1]

for i in range(5000):
    for j in range(4):
        x_test[4*i+j,:] = x1[i,:,0]
        y_test[4*i+j,:j+1] = y1[i,:j+1]

K.set_floatx('float32')
"""

#Looking at 50 steps in the past
input = Input(shape=(50,))

#Already predicted output by the Transformer
#If it's the first iteration then this will be [0,0,0,0]
predicted_output = Input(shape=(4,))

#Encoding
e = encoder(input)

#Decoding, using the encoded information and the predicted output
output = decoder(predicted_output, e)

#Creating the model
model = Model(inputs=[input,predicted_output],outputs=output)

#Cheking the model
model.summary()

#Checking the learning rate
lr = ReduceLROnPlateau(min_lr=0.00001,factor=0.2,patience=10)

#REMEBER: RADAM - optimizer that tries to eliminate 
#the need for ReduceLROnPlateau
model.compile(optimizer='Adam', loss='mse')

#Fitting the model
model.fit(y=y_train,x=x_train, batch_size=20, validation_data=(x_test,y_test),epochs=200,verbose=1,shuffle=1,callbacks=[lr])

#Saving the model
model.save('Heston_Transformer.h5')
