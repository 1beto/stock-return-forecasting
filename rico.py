import numpy as np
import h5py
import global_var
import random
import matplotlib.pyplot as plt

import keras
from keras.layers import Activation
from keras import backend, losses
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model

#Preparing the data
"""
N = global_var.N
M = global_var.M

data = h5py.File('stochastic_heston.h5','r')
data.require_dataset("Heston",((20*N**4),1001,2),dtype='float32')
shuffled_data = data["Heston"][...]
print(shuffled_data[1,...])
np.random.shuffle(shuffled_data)
print(shuffled_data[1,...])

# Dividindo o resultado entre os pontos à serem estudados e os pontos que devem
#ser adivinhados, como também os que devem ser treinados e os que devem ser testados
x_train,y_train = shuffled_data[:int(0.85*M),900:950,:], shuffled_data[:int(0.85*M),950:952,0]
x_test,y_test = shuffled_data[int(0.85*M):,900:950,:], shuffled_data[int(0.85*M):,950:952,0]
"""

#Preparing the neural network

keras.backend.set_floatx('float32')

model = Sequential()

#Entering analyzing 50 steps in the past, using the values of the
#return and the volatility
model.add(BatchNormalization(input_shape = (50,2)))

#Trasforming the input data in 1D
model.add(Flatten())

#Hidden Layers
model.add(Dense(
        units = 500,
        activation = 'relu',
        use_bias = True
        ))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(
        units = 300,
        activation = 'relu',
        use_bias = True
        ))
model.add(Dropout(0.2))

model.add(Dense(
        units = 100,
        activation = 'relu',
        use_bias = True
        ))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(
        units = 60,
        activation = 'relu',
        use_bias = True
        ))
model.add(Dropout(0.1))

model.add(Dense(
        units = 2,
        activation = 'linear',
        use_bias = True,
        ))

#model = load_model("HestonModel.h5")

#Checking the model
model.summary()

# %%

#Compiling the model
model.compile(loss = "mse", optimizer = "adam")

#Checking the model status with EarlyStopping and ReduceLROnPlateau
es = EarlyStopping(monitor="val_loss",mode="min",patience=50,)
lr = ReduceLROnPlateau(min_lr=0.00001,factor=0.2,patience=10)

#Fitting the model
model.fit(y=y_train,x=x_train, batch_size=20, validation_data=(x_test,y_test),epochs=200,verbose=2,shuffle=1,callbacks=[es,lr])

#Saving the model
model.save('HestonModel.h5')
