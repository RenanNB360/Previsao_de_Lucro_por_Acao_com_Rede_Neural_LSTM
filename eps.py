import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

print('\nCarregamento dos Dados')

df = pd.read_csv('dataset.csv')

eps = df['eps'].values.reshape(-1, 1)

def cria_dataset(data, look_back = 1):
    x, y = [], []

    for i in range(len(data) - look_back):
        a = data[i:(i + look_back), 0]
        x.append(a)
        y.append(data[i + look_back, 0])
    return np.array(x), np.array(y)

indice = int(len(eps) * 0.8)
train, test = eps[0: indice, :], eps[indice: len(eps), :]

scaler = MinMaxScaler(feature_range= (0, 1))

train_norm = scaler.fit_transform(train)
test_norm = scaler.transform(test)

look_back = 1
x_train, y_train = cria_dataset(train_norm, look_back)
x_test, y_test = cria_dataset(test_norm, look_back)

X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

modelo = tf.keras.models.Sequential([tf.keras.layers.LSTM(50, input_shape = (look_back, 1)),
                                     tf.keras.layers.Dense(1)])

modelo.compile(optimizer = 'adam', loss = 'mean_squared_error')

print('\nTreinamento do Modelo.\n')

modelo.fit(X_train, y_train, epochs = 50, batch_size = 1, verbose = 1)

previsao_train = modelo.predict(X_train)
previsao_test = modelo.predict(X_test)

previsao_train = scaler.inverse_transform(previsao_train)
y_train_rescaled  = scaler.inverse_transform([y_train])
previsao_test = scaler.inverse_transform(previsao_test)
y_test_rescaled = scaler.inverse_transform([y_test])

train_score = np.sqrt(mean_squared_error(y_test_rescaled[0], previsao_test[:, 0]))
print(f'\nRMSE em treino: {train_score:.2f}')

test_score = np.sqrt(mean_squared_error(y_test_rescaled[0], previsao_test[:, 0]))
print(f'RMSE em teste: {test_score:.2f}')

original_train_data_index = df['ano'][look_back:look_back + len(y_train_rescaled[0])]

original_test_data_index = df['ano'][len(y_train_rescaled[0]) + 2 * look_back:len(y_train_rescaled[0]) + 2 * look_back + len(y_test_rescaled[0])]

predicted_train_data_index = df['ano'][look_back:look_back + len(previsao_train)]

predicted_test_data_index = df['ano'][len(y_train_rescaled[0]) + 2 * look_back:len(y_train_rescaled[0]) + 2 * look_back+len(previsao_test)]

plt.figure(figsize=(15, 6))
plt.plot(original_train_data_index, y_train_rescaled[0], label="Dados de Treino Originais", color="blue", linestyle='-')
plt.plot(predicted_train_data_index, previsao_train[:, 0], label="Previsões em Treino", color="green", linestyle='--')
plt.plot(original_test_data_index, y_test_rescaled[0], label="Dados de Teste Originais", color="black", linestyle='-')
plt.plot(predicted_test_data_index, previsao_test[:, 0], label="Previsões em Teste", color="red", linestyle='--')
plt.title("EPS Real vs. EPS Previsto com IA")
plt.xlabel("Ano")
plt.ylabel("EPS")
plt.legend()
plt.grid(True)
plt.show()

last_data = test_norm[-look_back:]
last_data = np.reshape(last_data, (1, look_back, 1))

lista_previsoes = []

for _ in range(2):
    prediction = modelo.predict(last_data)
    lista_previsoes.append(prediction[0, 0])
    last_data = np.roll(last_data, shift = -1)
    last_data[0, look_back - 1, 0] = prediction

lista_previsoes_rescaled = scaler.inverse_transform(np.array(lista_previsoes).reshape(-1, 1))

print(f'\nPrevisão do EPS para 2024: {lista_previsoes_rescaled[0, 0]:.2f}')
print(f'Previsão do EPS ára 2025: {lista_previsoes_rescaled[1, 0]:.2f}')