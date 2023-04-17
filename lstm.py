import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from keras.losses import mean_squared_error
import matplotlib.pyplot as plt

# Carga los datos del archivo csv
dataset = pd.read_csv("airline-passengers.csv", usecols=[1], engine='python', skipfooter=3)

# Convierte los datos a un array de numpy
data = dataset.values.astype('float32')

# Normaliza los datos
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# Divide los datos en conjuntos de entrenamiento y prueba
train_size = int(len(data) * 0.67)
test_size = len(data) - train_size
train, test = data[0:train_size,:], data[train_size:len(data),:]

# Crea los conjuntos de datos de entrada/salida para el modelo
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Reshape de entrada para [muestras, tiempo, características]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Crea y entrena el modelo LSTM
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# Realiza las predicciones del modelo
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invierte la normalización de los datos para obtener valores reales
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Calcula el error cuadrático medio del modelo
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Puntuación del entrenamiento: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Puntuación del test: %.2f RMSE' % (testScore))

# Visualiza las predicciones del modelo junto con los datos reales
trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
testPredictPlot = np.empty_like(data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(data)-1, :] = testPredict
plt.plot(scaler.inverse_transform(data))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()