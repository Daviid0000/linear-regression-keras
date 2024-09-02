import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import SGD

datos = pd.read_csv('altura_peso.csv', sep=",", skiprows=32, usecols=[2, 3])
print(datos)

x = datos["altura"].values
y = datos["peso"] .values

np.random.seed(2)
modelo = Sequential()

input_dim = 1
output_dim = 1
modelo.add(Dense(output_dim, input_dim=input_dim, activation='linear'))

sgd = SGD(lr=0.0004)

modelo.compile(loss='mse', optimizer=sgd)

modelo.summary()