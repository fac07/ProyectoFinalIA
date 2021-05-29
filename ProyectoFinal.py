import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plt.style.use('fast')

from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
from sklearn.preprocessing import MinMaxScaler

from datetime import datetime
st.title("Reporte infectados Covid-19")
st.set_option('deprecation.showPyplotGlobalUse', False)

df = pd.read_csv('datosCovid19.csv',sep=";",  parse_dates=[0], header=None,index_col=0, squeeze=True,names=['fecha','infectados'])
st.write("Datos cargados")
st.write(df)

st.write("Header de los datos")
st.write(df.head())

st.write("Rango fechas cargadas")
st.write("**Inicio: **", df.index.min())
st.write("**Fin: **",df.index.max())

st.write("**Datos cargados 2020: **", len(df['2020']))
st.write("**Datos cargados 2021: **", len(df['2021']))

st.write("Datos estadisticos de carga")
st.write(df.describe())

meses = df.resample('M').mean()
st.write("Promedio de infectados por mes")
st.write(meses)

st.title("Comparación 2020-2021")

plt.plot(meses['2020'].values)
plt.plot(meses['2021'].values)
#plt.show()
st.pyplot()

st.title("Comparación meses más altos 2020-2021")
abril2020 = df['2020-07-01':'2020-07-31']
plt.plot(abril2020.values)
abril2021 = df['2021-04-01':'2021-04-30']
plt.plot(abril2021.values)
#plt.show()
st.pyplot()

PASOS = 8

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
# load dataset
values = df.values
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
values=values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
scaled = scaler.fit_transform(values)
# frame as supervised learning

st.write("Encabezado de datos transformados")
reframed = series_to_supervised(scaled, PASOS, 1)
st.write(reframed.head())

# split into train and test sets
values = reframed.values
n_train_days = 294+147 - (30+PASOS)
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# split into input and outputs
x_train, y_train = train[:, :-1], train[:, -1]
x_val, y_val = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
#st.write(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

def crear_modeloFF():
    model = Sequential() 
    model.add(Dense(PASOS, input_shape=(1,PASOS),activation='tanh'))
    model.add(Flatten())
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mean_absolute_error',optimizer='Adam',metrics=["mse"])
    model.summary()
    return model

EPOCHS=4000
model = crear_modeloFF()
history=model.fit(x_train,y_train,epochs=EPOCHS,validation_data=(x_val,y_val),batch_size=PASOS)

st.title("Validación de los datos últimos 30 días")
results=model.predict(x_val)
plt.scatter(range(len(y_val)),y_val,c='g')
plt.scatter(range(len(results)),results,c='r')
plt.title('validate')
st.pyplot()

st.title("Gráfica de perdida")
plt.ylim(0.12, 0.35)
plt.plot(history.history['loss'])
plt.title('loss')
plt.plot(history.history['val_loss'])
plt.title('validate loss')
st.pyplot()

st.title("Gráfica de errores")
plt.ylim(0.01, 0.18)
plt.title('Accuracy')
plt.plot(history.history['mse'])
st.pyplot()

st.write("Infectados últimos 15 días")
ultimosDias = df['2021-05-13':'2021-05-27']
st.write(ultimosDias)

values = ultimosDias.values
values = values.astype('float32')
# normalize features
values=values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, PASOS, 1)
reframed.drop(reframed.columns[[8]], axis=1, inplace=True)
st.write("Datos transformados")
st.write(reframed.head(8))

values = reframed.values
x_test = values[5:, :]
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
print(x_test)

def agregarNuevoValor(x_test,nuevoValor):
    for i in range(x_test.shape[2]-1):
        x_test[0][0][i] = x_test[0][0][i+1]
    x_test[0][0][x_test.shape[2]-1]=nuevoValor
    return x_test

results=[]
for i in range(8):
    parcial=model.predict(x_test)
    results.append(parcial[0])
    print(x_test)
    x_test=agregarNuevoValor(x_test,parcial[0])

adimen = [x for x in results]    
inverted = scaler.inverse_transform(adimen)

st.title("Proyección de datos")
st.write("Proyección siguientes 8 días")
st.write(inverted)

prediccion1SemanaDiciembre = pd.DataFrame(inverted)
prediccion1SemanaDiciembre.columns = ['pronostico']
prediccion1SemanaDiciembre.plot()
st.write("Gráfica de proyección")
st.pyplot()
#now = datetime.now() # current date and time

#year = now.strftime("%Y")
#month = now.strftime("%m")
#day = now.strftime("%d")
#time = now.strftime("%H%M%S")
#prediccion1SemanaDiciembre.to_csv('pronostico' + year+month+day+time +'.csv')
