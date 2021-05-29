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


st.title("Reporte proyección de población")
#
# Lectura y visualización del set de datos
#
DatosPaises = pd.read_excel('DatosMaestrosPaises.xlsx')
Paises = st.sidebar.selectbox('Elegir departamento', options = DatosPaises)
Anio = st.sidebar.number_input('Año para proyectar', min_value=2000, max_value=2999, value=2021, step=1)

if(st.sidebar.button("Cargar")):
    st.write("Datos cargados")
    datosExcel = pd.read_excel('OrigenDatosPoblacionPaises.xlsx')
    datos = pd.DataFrame()
    datos['Año'] = datosExcel['Año']
    datos[Paises] = datosExcel[Paises]
    st.write(datos)

    # Al graficar los datos se observa una tendencia lineal
    st.set_option('deprecation.showPyplotGlobalUse', False)
    datos.plot.scatter(x='Año', y=Paises)
    plt.xlabel('Año')
    plt.ylabel('Población')
    #plt.show()
    st.write("Gráfica de datos")
    st.pyplot()

    x = datos['Año'].values
    y = datos[Paises].values

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

    sgd = SGD(lr=0.004)
    modelo.compile(loss='mse', optimizer=sgd)

    # Imprimir en pantalla la información del modelo
    modelo.summary()
    #
    # Entrenamiento: realizar la regresión lineal
    #

    # 40000 iteraciones y todos los datos de entrenamiento (29) se usarán en cada
    # iteración (batch_size = 29)

    num_epochs = 1
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
    x_pred = np.array([Anio])
    y_pred = modelo.predict(x_pred)
    st.write(y_pred)
    st.write("La población sera de {} mm-Hg".format(y_pred[0][0]), " para el año {} ".format(x_pred[0]))