import numpy as np
import h5py
import global_var
import random
import matplotlib.pyplot as plt

#Raiz quarta do número de simulações à serem feitas
N = global_var.N
M = global_var.M
#Abrindo arquivo com os dados

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

# %%
import keras
from keras.layers import Activation
from keras import backend
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Dense,Dropout,Flatten
from keras.models import Sequential
from keras import losses
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from scipy.optimize import differential_evolution
from keras.models import load_model

keras.backend.set_floatx('float32')

model = Sequential()

model.add(BatchNormalization(input_shape = (50,2)))

model.add(Flatten())

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
model.summary()

# %%


model.compile(loss = "mae", optimizer = "adam")

es = EarlyStopping(monitor="val_loss",mode="min",patience=300,)
lr = ReduceLROnPlateau(min_lr=0.00001,factor=0.2,patience=10)

model.fit(y=y_train,x=x_train, batch_size=20, validation_data=(x_test,y_test),epochs=200,verbose=2,shuffle=1,callbacks=[es,lr])

model.save('HestonModel.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# ## Store/Load optimal NN parameteres
# %%

np.savetxt("y_test.dat",y_test)
np.savetxt("x_test.dat",x_test)
# %%
