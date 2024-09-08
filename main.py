import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import SGD

# obtengo los datos de archivo excel
datos = pd.read_csv('altura_peso.csv', sep=",")
print(datos)

# obtengo los datos de cada columna del archivo excel
x = datos["Altura"].values
y = datos["Peso"].values

np.random.seed(2)

# inicializo un modelo con Sequential
modelo = Sequential()

# declado que habrá un valor de entrada y uno de salida
input_dim = 1
output_dim = 1

modelo.add(Dense(output_dim, input_dim=input_dim, activation='linear'))

# añado la tasa de aprendizaje expresando en la notación exponencial 1e-5 esto equivale a poner 0.00001 (no se ha usado 0.0004 porque da nan en las pérdiads)
sgd = SGD(learning_rate=0.00001)

modelo.compile(loss='mse', optimizer=sgd)

modelo.summary()

# defino la cantidad de entrenamientos que hará el modelo
num_epochs = 10000
batch_size = 50
historia = modelo.fit(x, y, epochs=num_epochs, batch_size=batch_size,
verbose=1)

capas = modelo.layers[0]
w, b = capas.get_weights()
print('Parámetros: w = {:.1f}, b = {:.1f}'.format(w[0][0],b[0]))

# utilizo la librería de matplotlib parar graficar las pérdidas y la regreseion lineal
plt.subplot(1,2,1)
plt.plot(historia.history["loss"])
plt.xlabel('epoch')
plt.ylabel('ECM')
plt.title('ECM vs. epochs')
y_regr = modelo.predict(x)
plt.subplot(1, 2, 2)
plt.scatter(x,y)
plt.plot(x,y_regr,'r')
plt.title('Datos originales y regresión lineal')
plt.show()

# hago una predicción sobre el peso de una persona que mide 1,55mts
x_pred = np.array([170])
y_pred = modelo.predict(x_pred)
print("El peso será de {:.1f} kg".format(y_pred[0][0]), "para una persona de {} cm".format(x_pred[0]))