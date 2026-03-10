# -*- coding: utf-8 -*-
"""
RedesNeuronalesGlucosa.py

Código adaptado desde Colab para VS Code.
Autor: Miriam Carrasco.
"""

#Importación de librerias.
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
import seaborn as sns
from scipy.io import savemat

# -------------------------------
# 1. Fijar la mejor semilla
# -------------------------------

import os, random, numpy as np, tensorflow as tf
os.environ["PYTHONHASHSEED"] = "10"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
random.seed(10)
np.random.seed(10)
tf.random.set_seed(10)

mejor_semilla = 10
tf.random.set_seed(mejor_semilla)
np.random.seed(mejor_semilla)
random.seed(mejor_semilla)
#Carga de base de datos
datos = pd.read_csv('Glucosa_Data_Base_Clean.csv')
print(datos.info())
print(datos.head())

#Limpieza de datos
datos_clean=datos.dropna()
print(datos_clean.info())
print(datos_clean.head())


# -------------------------------
# 2. Separar datos
# -------------------------------
X = datos_clean.drop('Glucosa', axis=1) # Todas las columnas
y = datos_clean['Glucosa'] #Columna de glucosa

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=mejor_semilla)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=mejor_semilla)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_val_std = scaler.transform(X_val)
X_test_std = scaler.transform(X_test)

# -------------------------------
# 3. Definir el modelo con los mejores parámetros
# (ejemplo: hidden1=64, hidden2=32, dropout=0.3, l2=0.001, lr=0.001, epochs=200, batch_size=15)
# -------------------------------
final_model = Sequential([
        Dense(128, activation='sigmoid', input_shape=(X_train_std.shape[1],), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='sigmoid', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
optimizer = Adam(learning_rate=0.001)
final_model.compile(optimizer=optimizer, loss='mse', metrics=['mae','mse'])

# -------------------------------
# 4. Entrenar con los mejores parámetros
# -------------------------------
final_model.fit(
    X_train_std, y_train,
    validation_data=(X_val_std, y_val),
    epochs=200,
    batch_size=15,
    verbose=1
)

# -------------------------------
# 5. Evaluar en test
# -------------------------------
y_pred = final_model.predict(X_test_std).flatten()
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)
pearson, _ = pearsonr(y_test, y_pred)

print(f"📊 Evaluación final: MAE={mae:.4f}, MSE={mse:.4f}, R2={R2:.4f}, Pearson={pearson:.4f}")

# -------------------------------
# 6. Guardar el modelo entrenado
# -------------------------------
final_model.save("modelo_glucosa_mejor.h5")
print("✅ Modelo guardado en 'modelo_glucosa_mejor.h5'")



# Predicciones en cada conjunto
y_train_pred = final_model.predict(X_train_std).flatten()
y_val_pred   = final_model.predict(X_val_std).flatten()
y_test_pred  = final_model.predict(X_test_std).flatten()

# Función para graficar regresión
def plot_regression(y_true, y_pred, filename):
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, edgecolor="k")
    # Línea de identidad
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', lw=2, label="Línea identidad")
    plt.xlabel("Glucosa de referencia [mg/dL]")
    plt.ylabel("Predicciones")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()

# Graficar cada conjunto
plot_regression(y_train, y_train_pred,"regresion_train.pdf")
plot_regression(y_val, y_val_pred,"regresion_val.pdf")
plot_regression(y_test, y_test_pred,"regresion_test-pdf")


#Generar predicciones para train, val y test
y_train_pred = final_model.predict(X_train_std).flatten()
y_val_pred = final_model.predict(X_val_std).flatten()
y_test_pred = final_model.predict(X_test_std).flatten()


def print_metrics(y_true, y_pred, set_name):
  mae = mean_absolute_error(y_true, y_pred)
  mse = mean_squared_error(y_true, y_pred)
  R2 = r2_score(y_true, y_pred)
  r_value, _ = pearsonr(y_true, y_pred)
  print(f"{set_name}:")
  print(f"  MAE: {mae:.4f}")
  print(f"  MSE: {mse:.4f}")
  print(f"  R2: {R2:.4f}")
  print(f"  R: {r_value:.4f}")


print_metrics(y_train, y_train_pred, "Entrenamiento")
print_metrics(y_val, y_val_pred, "Validación")
print_metrics(y_test, y_test_pred, "Prueba")



# Crear DataFrame con valores reales y predicciones
df_resultados = pd.DataFrame({
    "y_test": y_test.values,   # valores reales
    "y_pred": y_pred           # predicciones del modelo
})

# Guardar en CSV (fácil de abrir en MATLAB con readtable)
df_resultados.to_csv("resultados_test.csv", index=False)

# Guardar en formato MAT (nativo de MATLAB)
savemat("resultados_test.mat", {
    "y_test": y_test.values,
    "y_pred": y_pred
})



#Generación de gráfica de norma ISO 15197:2015

x = y_test  # Glucosa medida [mg/dL]
y = y_test - y_test_pred  # Diferencia estimado - medido [mg/dL]
val_max = np.max(x)

# Crear figura y ejes
fig, ax = plt.subplots(figsize=(6, 6))

# Gráfica de dispersión con estilo
ax.scatter(x, y,
           color='royalblue',
           edgecolor='black',
           alpha=0.7,
           s=60,
           label='Predicciones')

# Límites ISO 15197:2015
x = np.linspace(0, 200, 200)

# Para glucosa < 100 mg/dL → ±15 mg/dL
ax.plot([0, 100], [15, 15], 'r--', linewidth=1.5)
ax.plot([0, 100], [-15, -15], 'r--', linewidth=1.5)

# Para glucosa ≥ 100 mg/dL → ±15% del valor medido
ax.plot([100, 200], [15, 200*0.15], 'r--', linewidth=1.5)
ax.plot([100, 200], [-15, -200*0.15], 'r--', linewidth=1.5)

# Ajustes de aspecto
ax.set_xlim(0, 200)
ax.set_ylim(-40, 40)  # rango de diferencias
ax.set_xlabel('Glucosa de referencia [mg/dL]', fontsize=12)
ax.set_ylabel('Diferencia estimado - medido [mg/dL]', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("norma_iso_test.pdf", format="pdf", bbox_inches="tight")
plt.show()


# Gráfica de histograma de diferencias
plt.figure(figsize=(7,5))
plt.hist(diff_values, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("Diferencia (Real - Predicción)")
plt.ylabel("Frecuencia")
plt.grid(True, linestyle="--", alpha=0.7)

plt.savefig("histograma_diferencias1.pdf", format="pdf", bbox_inches="tight")
plt.show()


# Test de Shapiro-Wilk
stat, p = stats.shapiro(diff_values)
print("Estadístico Shapiro-Wilk:", stat)
print("p-value:", p)

if p > 0.05:
    print("No se rechaza la normalidad (las diferencias son aproximadamente normales).")
else:
    print("Se rechaza la normalidad (las diferencias no siguen distribución normal).")

#Generación de gráfica de Bland-Altman para el set de prueba

# Valores reales y predicciones del set de prueba
y_true = y_test.values
y_pred = y_pred

# Calcular medias y diferencias
mean_values = np.mean([y_true, y_pred], axis=0)
diff_values = y_true - y_pred

# Estadísticos
mean_diff = np.mean(diff_values)  # sesgo
std_diff = np.std(diff_values, ddof=1)  # desviación estándar
loa_upper = mean_diff + 1.96 * std_diff  # límite superior
loa_lower = mean_diff - 1.96 * std_diff  # límite inferior

# Gráfico Bland-Altman
plt.figure(figsize=(7,6))
plt.scatter(mean_values, diff_values, alpha=0.6, edgecolor="k")
plt.axhline(mean_diff, color='red', linestyle='--', label=f"Sesgo = {mean_diff:.2f}")
plt.axhline(loa_upper, color='blue', linestyle='--', label=f"Límite superior = {loa_upper:.2f}")
plt.axhline(loa_lower, color='blue', linestyle='--', label=f"Límite inferior = {loa_lower:.2f}")
plt.xlabel("Media entre valores reales y predicciones")
plt.ylabel("Diferencia (Real - Predicción)")
#pt.title("Gráfico de Bland-Altman (Test Set)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.ylim(-40,40)
plt.savefig("bland_altman_test.pdf", format="pdf", bbox_inches="tight")
plt.show()
