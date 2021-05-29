import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense
#from tensorflow.keras.layers.core import Dense
from tensorflow.keras.optimizers import SGD


st.title("Reporte mortalidad")
#
# Lectura y visualización del set de datos
#

st.write("Datos maestros")
st.write("Rangos de edad")
datosRangos = pd.read_excel('DatoMaestroRangosEdad.xlsx')
st.write(datosRangos)

st.write("Enfermedades")
datosEnfermedades = pd.read_excel('DatoMaestroEnfermedades.xlsx')
st.write(datosEnfermedades)

st.write("Datos cargados")
datosExcel = pd.read_excel('OrigenDatosMortalidad.xlsx')

st.write(datosExcel)
datosOperados = pd.DataFrame()
datosOperados['No.'] = datosExcel['No.']
datosOperados['Diagnóstico'] = datosExcel['Diagnóstico']

datosOperados['< 1 mes F'], datosOperados['< 1 mes M'] = datosExcel['< 1 mes F'], datosExcel['< 1 mes M']
datosOperados['< 1 mes T'] = datosExcel['< 1 mes F'] + datosExcel['< 1 mes M']


datosOperados['1m a < 2m F'], datosOperados['1m a < 2m M'] = datosExcel['1m a < 2m F'], datosExcel['1m a < 2m M']
datosOperados['1m a < 2m T'] = datosExcel['1m a < 2m F'] + datosExcel['1m a < 2m M']


datosOperados['2m a < 1 año F'], datosOperados['2m a < 1 año M'] = datosExcel['2m a < 1 año F'], datosExcel['2m a < 1 año M']
datosOperados['2m a < 1 año T'] = datosExcel['2m a < 1 año F'] + datosExcel['2m a < 1 año M']


datosOperados['1a a 4 años F'], datosOperados['1a a 4 años M'] = datosExcel['1a a 4 años F'], datosExcel['1a a 4 años M']
datosOperados['1a a 4 años T'] = datosExcel['1a a 4 años F'] + datosExcel['1a a 4 años M']


datosOperados['5 a 9 años F'], datosOperados['5 a 9 años M'] = datosExcel['5 a 9 años F'], datosExcel['5 a 9 años M']
datosOperados['5 a 9 años T'] = datosExcel['5 a 9 años F'] + datosExcel['5 a 9 años M']


datosOperados['10 a 14 años F'], datosOperados['10 a 14 años M'] = datosExcel['10 a 14 años F'], datosExcel['10 a 14 años M']
datosOperados['10 a 14 años T'] = datosExcel['10 a 14 años F'] + datosExcel['10 a 14 años M']


datosOperados['15 a 19 años F'], datosOperados['15 a 19 años M'] = datosExcel['15 a 19 años F'], datosExcel['15 a 19 años M']
datosOperados['15 a 19 años T'] = datosExcel['15 a 19 años F'] + datosExcel['15 a 19 años M']


datosOperados['20 a 24 años F'], datosOperados['20 a 24 años M'] = datosExcel['20 a 24 años F'], datosExcel['20 a 24 años M']
datosOperados['20 a 24 años T'] = datosExcel['20 a 24 años F'] + datosExcel['20 a 24 años M']

datosOperados['25 a 29 años F'], datosOperados['25 a 29 años M'] = datosExcel['25 a 29 años F'], datosExcel['25 a 29 años M']
datosOperados['25 a 29 años T'] = datosExcel['25 a 29 años F'] + datosExcel['25 a 29 años M']


datosOperados['30 a 34 años F'], datosOperados['30 a 34 años M'] = datosExcel['30 a 34 años F'], datosExcel['30 a 34 años M']
datosOperados['30 a 34 años T'] = datosExcel['30 a 34 años F'] + datosExcel['30 a 34 años M']


datosOperados['35 a 39 años F'], datosOperados['35 a 39 años M'] = datosExcel['35 a 39 años F'], datosExcel['35 a 39 años M']
datosOperados['35 a 39 años T'] = datosExcel['35 a 39 años F'] + datosExcel['35 a 39 años M']


datosOperados['40 a 44 años F'], datosOperados['40 a 44 años M'] = datosExcel['40 a 44 años F'], datosExcel['40 a 44 años M']
datosOperados['40 a 44 años T'] = datosExcel['40 a 44 años F'] + datosExcel['40 a 44 años M']


datosOperados['45 a 49 años F'], datosOperados['45 a 49 años M'] = datosExcel['45 a 49 años F'], datosExcel['45 a 49 años M']
datosOperados['45 a 49 años T'] = datosExcel['45 a 49 años F'] + datosExcel['45 a 49 años M']


datosOperados['50 a 54 años F'], datosOperados['50 a 54 años M'] = datosExcel['50 a 54 años F'], datosExcel['50 a 54 años M']
datosOperados['50 a 54 años T'] = datosExcel['50 a 54 años F'] + datosExcel['50 a 54 años M']


datosOperados['55 a 59 años F'], datosOperados['55 a 59 años M'] = datosExcel['55 a 59 años F'], datosExcel['55 a 59 años M']
datosOperados['55 a 59 años T'] = datosExcel['55 a 59 años F'] + datosExcel['55 a 59 años M']


datosOperados['60 a 64 años F'], datosOperados['60 a 64 años M'] = datosExcel['60 a 64 años F'], datosExcel['60 a 64 años M']
datosOperados['60 a 64 años T'] = datosExcel['60 a 64 años F'] + datosExcel['60 a 64 años M']


datosOperados['65 a 69 años F'], datosOperados['65 a 69 años M'] = datosExcel['65 a 69 años F'], datosExcel['65 a 69 años M']
datosOperados['65 a 69 años T'] = datosExcel['65 a 69 años F'] + datosExcel['65 a 69 años M']


datosOperados['70+ F'], datosOperados['70+ M'] = datosExcel['70+ F'], datosExcel['70+ M']
datosOperados['70+ T'] = datosExcel['70+ F'] + datosExcel['70+ M']

st.write("Datos operados")
st.write(datosOperados)

datos = pd.DataFrame([])

data = []
for index, row in datosOperados.iterrows():
    for i in range(row['< 1 mes T']):
        data.append(dict([('Enfermedad',row['No.']),('Rango de edad', 1)]))
    for i in range(row['1m a < 2m T']):
        data.append(dict([('Enfermedad',row['No.']),('Rango de edad', 2)]))
    for i in range(row['2m a < 1 año T']):
        data.append(dict([('Enfermedad',row['No.']),('Rango de edad', 3)]))
    for i in range(row['1a a 4 años T']):
        data.append(dict([('Enfermedad',row['No.']),('Rango de edad', 4)]))
    for i in range(row['5 a 9 años T']):
        data.append(dict([('Enfermedad',row['No.']),('Rango de edad', 5)]))
    for i in range(row['10 a 14 años T']):
        data.append(dict([('Enfermedad',row['No.']),('Rango de edad', 6)]))
    for i in range(row['15 a 19 años T']):
        data.append(dict([('Enfermedad',row['No.']),('Rango de edad', 7)]))
    for i in range(row['20 a 24 años T']):
        data.append(dict([('Enfermedad',row['No.']),('Rango de edad', 8)]))
    for i in range(row['25 a 29 años T']):
        data.append(dict([('Enfermedad',row['No.']),('Rango de edad', 9)]))
    for i in range(row['30 a 34 años T']):
        data.append(dict([('Enfermedad',row['No.']),('Rango de edad', 10)]))
    for i in range(row['35 a 39 años T']):
        data.append(dict([('Enfermedad',row['No.']),('Rango de edad', 11)]))
    for i in range(row['40 a 44 años T']):
        data.append(dict([('Enfermedad',row['No.']),('Rango de edad', 12)]))
    for i in range(row['45 a 49 años T']):
        data.append(dict([('Enfermedad',row['No.']),('Rango de edad', 13)]))
    for i in range(row['50 a 54 años T']):
        data.append(dict([('Enfermedad',row['No.']),('Rango de edad', 14)]))
    for i in range(row['55 a 59 años T']):
        data.append(dict([('Enfermedad',row['No.']),('Rango de edad', 15)]))
    for i in range(row['60 a 64 años T']):
        data.append(dict([('Enfermedad',row['No.']),('Rango de edad', 16)]))
    for i in range(row['65 a 69 años T']):
        data.append(dict([('Enfermedad',row['No.']),('Rango de edad', 17)]))
    for i in range(row['70+ T']):
        data.append(dict([('Enfermedad',row['No.']),('Rango de edad', 18)]))
datos = pd.DataFrame(data)

st.write("Datos tabulados")
st.write(datos)

# Al graficar los datos se observa una tendencia lineal
fig, ax = plt.subplots()
st.set_option('deprecation.showPyplotGlobalUse', False)
datos.plot.scatter(x='Rango de edad', y='Enfermedad')
plt.xlabel('Rangos de edades')
plt.ylabel('Enfermedades')
plt.show()
st.write("Gráfica de datos")
st.pyplot()

x = datos['Rango de edad'].values
y = datos['Enfermedad'].values

#
# Construir el modelo en Keras
#

# - Capa de entrada: 1 dato (cada dato "x" correspondiente a la edad)
# - Capa de salida: 1 dato (cada dato "y" correspondiente a la regresión lineal)
# - Activación: 'linear' (pues se está implementando la regresión lineal)

np.random.seed(2)			# Para reproducibilidad del entrenamiento

input_dim = 1
output_dim = 1
modelo = Sequential()
modelo.add(Dense(output_dim, input_dim=input_dim, activation='linear'))

# Definición del método de optimización (gradiente descendiente), con una
# tasa de aprendizaje de 0.0004 y una pérdida igual al error cuadrático
# medio

sgd = SGD(lr=0.0004)
modelo.compile(loss='mse', optimizer=sgd)

# Imprimir en pantalla la información del modelo
modelo.summary()
#
# Entrenamiento: realizar la regresión lineal
#

# 40000 iteraciones y todos los datos de entrenamiento (29) se usarán en cada
# iteración (batch_size = 29)

num_epochs = 40000
batch_size = x.shape[0]
history = modelo.fit(x, y, epochs=num_epochs, batch_size=batch_size, verbose=0)

#
# Visualizar resultados del entrenamiento
#

# Imprimir los coeficientes "w" y "b"
capas = modelo.layers[0]
w, b = capas.get_weights()
st.write('Parámetros: w = {:.1f}, b = {:.1f}'.format(w[0][0],b[0]))

# Graficar el error vs epochs y el resultado de la regresión
# superpuesto a los datos originales
plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('ECM')
plt.title('ECM vs. epochs')

y_regr = modelo.predict(x)
plt.subplot(1, 2, 2)
plt.scatter(x,y)
plt.plot(x,y_regr,'r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Datos originales y regresión lineal')
# plt.show()
st.pyplot()

# Predicción
x_pred = np.array([1])
y_pred = modelo.predict(x_pred)
st.write("La presión sanguínea será de {} mm-Hg".format(y_pred[0][0]), " para una persona de {} años".format(x_pred[0]))