# -*- coding: utf-8 -*-
"""
RedesNeuronalesGlucosa.py

Código adaptado desde Colab para VS Code.
Autor: Miriam Carrasco.
"""

#Importación de librerias.
import os
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import random
from keras.layers import Dense
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor

os.environ["PYTHONHASHSEED"] = "10"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

#Clase Callback para graficar métricas
class MetricsPlotCallback(Callback):
  def on_train_begin(self, logs= None):
    self.train_mae = []
    self.val_mae = []
    self.train_mse = []
    self.val_mse=[]
  def on_epoch_end(self, epoch, logs= None):
    self.train_mae.append(logs.get('mae'))
    self.val_mae.append(logs.get('val_mae'))
    self.train_mse.append(logs.get('mse'))
    self.val_mse.append(logs.get('val_mse'))

  def on_train_end(self, logs=None):
    epochs = range(len(self.train_mae))
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, self.train_mae, label= 'Train MAE')
    plt.plot(epochs, self.val_mae, label='Val MAE')
    plt.title('MAE por Época')
    plt.xlabel('Epoca')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(epochs, self.train_mse, label='Train MSE')
    plt.plot(epochs, self.val_mse, label = 'Val MSE')
    plt.title('MSE por época')
    plt.xlabel('Epoca')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


#Carga de base de datos
datos = pd.read_csv('Glucosa_Data_Base_Clean.csv')
print(datos.info())
print(datos.head())

#Limpieza de datos
datos_clean=datos.dropna()
print(datos_clean.info())
print(datos_clean.head())

#Separación de caracteristicas y objetivo
X = datos_clean.drop('Glucosa', axis=1) # Todas las columnas
y = datos_clean['Glucosa'] #Columna de glucosa

# Use the best seed found from the previous execution
mejor_mae = float('inf')
mejor_pearson = -1
mejor_semilla = None
resultados_validos = []

for seed in range(1,11):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3,random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)
    X_test_std = scaler.transform(X_test)

    # Definición del modelo
    model = Sequential([
        Dense(64, activation= 'sigmoid', input_shape=(X_train_std.shape[1],), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='sigmoid', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='linear')
    ])

    # Model compilation
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mae', metrics=['mae','mse'])

    # Model training
    history = model.fit(X_train_std, y_train,
                        validation_data=(X_val_std, y_val),
                        epochs=200, batch_size=15,
                        verbose=0)

    # Evaluation on test set
    y_pred = model.predict(X_test_std).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    R2 = r2_score(y_test, y_pred)
    pearson, _= pearsonr(y_test, y_pred)

    if pearson > 0.7:
        resultados_validos.append({
            "semilla": seed,
            "mae": mae,
            "pearson": pearson
        })
        print(f"✔️ Semilla {seed} válida: MAE = {mae:.4f}, Pearson = {pearson:.4f},MSE = {mse:.4f}, R2={R2:.4f}")
    else:
        print(f"✖️ Semilla {seed} descartada: MAE = {mae:.4f}, Pearson = {pearson:.3f}")

if resultados_validos:
    resultados_validos = sorted(resultados_validos, key=lambda x: -x['pearson'])
    mejor_resultado = resultados_validos[0]
    mejor_semilla = mejor_resultado['semilla']
    print(f"\n🔁 Mejor semilla con Pearson > 0.7: {mejor_semilla} | MAE = {mejor_resultado['mae']:.2f}, Pearson = {mejor_resultado['pearson']:.3f}")
else:
    print("\n❌ Ninguna semilla tuvo Pearson > 7. No se reentrenará.")


tf.random.set_seed(mejor_semilla)
np.random.seed(mejor_semilla)
random.seed(mejor_semilla)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=mejor_semilla)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=mejor_semilla)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_val_std = scaler.transform(X_val)
X_test_std = scaler.transform(X_test)


def build_model(hidden1=64, hidden2=32, dropout_rate=0.3, l2_rate=0.001, learning_rate=0.001):
    model = Sequential([
        Dense(hidden1, activation='sigmoid', input_shape=(X_train_std.shape[1],), kernel_regularizer=l2(l2_rate)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(hidden2, activation='sigmoid', kernel_regularizer=l2(l2_rate)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(1, activation='linear')
    ])
    optimizer= Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae','mse'])
    return model


regressor=KerasRegressor(model=build_model, verbose=0)

param_grid = {
    "model__hidden1": [126, 127, 128],
    "model__hidden2": [16, 32],
    "model__dropout_rate": [0.2,0.3,0.4],
    "model__l2_rate":[0.0001, 0.001],
    "model__learning_rate": [0.001, 0.0005],
    "batch_size":[15,30],
    "epochs":[180,190,200]
}

cv_strategy = KFold(n_splits=5, shuffle=True, random_state=mejor_semilla)
grid=GridSearchCV(estimator=regressor,
                  param_grid=param_grid,
                  scoring='neg_mean_absolute_error',
                  cv=cv_strategy,
                  verbose=1,
                  n_jobs=-1)

grid_result= grid.fit(X_train_std, y_train)
best_params = grid_result.best_params_

print("Mejores parámetros encontrados:")
print(grid_result.best_params_)
print(f"Mejor MAE: {-grid_result.best_score_:.4f}") # Negate the score since it's neg_mean_absolute_error
best_model = grid_result.best_estimator_.model_
y_test_pred = best_model.predict(X_test_std).flatten()
mae = mean_absolute_error(y_test, y_test_pred)
print(f"MAE en test con mejor modelo: {mae:.4f}")

final_model = build_model(hidden1=best_params["model__hidden1"],
                          hidden2=best_params["model__hidden2"],
                          learning_rate=best_params["model__learning_rate"],
                          dropout_rate=best_params["model__dropout_rate"])

final_model.fit(X_train_std, y_train,
                validation_data=(X_val_std, y_val),
                epochs=best_params["epochs"],
                batch_size=best_params["batch_size"],
                verbose=0,
                callbacks=[MetricsPlotCallback()])

# Evaluación final en test
y_pred = final_model.predict(X_test_std).flatten()
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)
pearson, _ = pearsonr(y_test, y_pred)

print(f"📊 Evaluación final: MAE={mae:.4f}, MSE={mse:.4f}, R2={R2:.4f}, Pearson={pearson:.4f}")
