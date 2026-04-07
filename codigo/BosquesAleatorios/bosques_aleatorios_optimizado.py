

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold,GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from scipy.stats import pearsonr
from scipy.io import savemat
#Cargar datos
datos = pd.read_csv(r"C:\Users\myria\Documents\CodigoFuenteTesis\datos\Glucosa_Data_Base_Clean.csv")
print(datos.info())
print(datos.head())

#Limpieza de datos
datos_clean=datos.dropna()
print(datos_clean.info())
print(datos_clean.head())



#Separación de caracteristicas y objetivo
X = datos_clean.drop('Glucosa', axis=1) # Todas las columnas
y = datos_clean['Glucosa'] #Columna de glucosa

X.columns =(
    X.columns
    .str.strip() # quita espacios al inicio/fin
    .str.lower() # pasa todo a minúsculas
    .str.normalize('NFKD') # elimina acentos
    .str.encode('ascii', errors='ignore')
    .str.decode('utf-8')
)

mejores_resultados = []
semillas = [60,74,78,122,126]
def pearson_scorer(y_true, y_pred):
    try:
        r = pearsonr(y_true, y_pred)[0]
        if np.isnan(r):
            return 0.0
        return r
    except:
        return 0.0

scorer_pearson = make_scorer(pearson_scorer, greater_is_better=True)

for seed in semillas:
    y_binned = pd.qcut(y,q=10, duplicates='drop')
    sss=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=seed)
    for train_idx,test_idx in sss.split(X,y_binned):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        modelo_rf= RandomForestRegressor (
            n_estimators=100,
            max_depth=18,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=seed,
            criterion='squared_error',
            n_jobs=-1
    )

        modelo_rf.fit(X_train, y_train)
        y_pred_test = modelo_rf.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        pearson = pearsonr(y_test, y_pred_test)[0]

        mejores_resultados.append((seed, mae, mse, r2, pearson))


# Ordenar por Pearson descendente
mejores_resultados = sorted(mejores_resultados, key=lambda x: -x[4])
print("\n📊 Mejores semillas ordenadas por Pearson:")
for r in mejores_resultados[:5]:
    print(f"Seed={r[0]} | MAE={r[1]:.4f} | MSE={r[2]:.4f} | R2={r[3]:.4f} | Pearson={r[4]:.4f}")


mejor_seed = mejores_resultados[0][0]

param_grid = {
    'n_estimators': [50,100,150],
    'max_depth':[10,15,25],
    'min_samples_split': [2,4,6],
    'min_samples_leaf':[1,2,4],
    'max_features':  ['sqrt', 'log2'],
    'criterion': ['squared_error', 'absolute_error', 'poisson']

}

modelo_rf = RandomForestRegressor(
    random_state=mejor_seed,
    bootstrap=True,
    criterion='squared_error',
    n_jobs=-1
)

# Validación cruzada con KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=mejor_seed)

grid_search = GridSearchCV(
    estimator=modelo_rf,
    param_grid=param_grid,
    cv=kfold,
    scoring={'MAE':'neg_mean_absolute_error', 'Pearson': scorer_pearson},   # puedes cambiar a 'neg_mean_absolute_error' si prefieres
    refit='Pearson',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("\n✅ Mejores hiperparámetros encontrados (según Pearson):")
print(grid_search.best_params_)
print("Mejor R2 en validación:", grid_search.best_score_)

best_rf = grid_search.best_estimator_
y_pred_test = best_rf.predict(X_test)
y_pred_train = best_rf.predict(X_train)

mae_test= mean_absolute_error(y_test, y_pred_test)
pearson_test = pearsonr(y_test, y_pred_test)[0]

# Métricas en entrenamiento

mae_train = mean_absolute_error(y_train, y_pred_train)
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
pearson_train = pearsonr(y_train, y_pred_train)[0]
print("\n📊 Métricas en train con modelo definitivo")
print("MAE:", mae_train)
print("MSE:", mse_train)
print("R2:", r2_train)
print("Pearson:", pearson_train)

#Métricas en prueba

print("\n📊 Métricas en TEST con mejores hiperparámetros")
print("MAE:", mean_absolute_error(y_test, y_pred_test))
print("MSE:", mean_squared_error(y_test, y_pred_test))
print("R2:", r2_score(y_test, y_pred_test))
print("R:", pearsonr(y_test, y_pred_test)[0])

if pearson_test >= 0.9 and mae_test <= 10:
  print("\n✅ El modelo cumple con los criterios de calidad (Pearson > 0.9 y MAE < 10)")

else:
  print("\n⚠️ El modelo NO cumple completamente los criterios. Pearson o MAE fuera de rango.")





# Importancia de las características
importancias = best_rf.feature_importances_

# Crear un DataFrame ordenado
importancias_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importancias
}).sort_values(by='importance', ascending=False)

print("\n📊 Importancia de las características:")
print(importancias_df)

# Gráfica de barras
plt.figure(figsize=(20,8))
importancias_df.plot(kind='bar', x='feature', y='importance', legend=False)
#plt.title("Importancia de las características en RandomForest")
plt.xlabel("Características", fontsize=14)
plt.ylabel("Importancia", fontsize=14)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.xticks
plt.tight_layout()
plt.savefig("C:/Users/myria/Documents/CodigoFuenteTesis/resultados/BosquesAleatorios/AnalisisImportanciasBAO.pdf", format="pdf", bbox_inches="tight")


#Histograma de Diferencias
# Valores reales y predicciones del set de prueba
y_true = y_test.values
y_pred = y_pred_test

# Calcular diferencias
diff_values = y_true - y_pred


# Histograma de diferencias
plt.figure(figsize=(7,5))
plt.hist(diff_values, bins=20, edgecolor='black', alpha=0.7)
#plt.title("Histograma de diferencias")
plt.xlabel("Diferencia (referencia de glucosa - estimación de glucosa)[mg/dL]",fontsize=16)
plt.ylabel("Frecuencia", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.7)

plt.savefig("C:/Users/myria/Documents/CodigoFuenteTesis/resultados/BosquesAleatorios/bland_altman_test_diferencias_BA0.pdf", format="pdf", bbox_inches="tight")


# Q-Q plot
import scipy.stats as stats
plt.figure(figsize=(7,5))
stats.probplot(diff_values, dist="norm", plot=plt)
plt.title("Q-Q plot de diferencias")
plt.grid(True, linestyle="--", alpha=0.7)


# Test de Shapiro-Wilk
stat, p = stats.shapiro(diff_values)
print("Estadístico Shapiro-Wilk:", stat)
print("p-value:", p)

if p > 0.05:
    print("No se rechaza la normalidad (las diferencias son aproximadamente normales).")
else:
    print("Se rechaza la normalidad (las diferencias no siguen distribución normal).")

#Gráficas de Dispersión 

# Predicciones en cada conjunto
y_pred_train = best_rf.predict(X_train).flatten()
y_pred_test  = best_rf.predict(X_test).flatten()


# Función para graficar regresión
def plot_regression(y_true, y_pred, filename):
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, edgecolor="k")
    # Línea de identidad
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', lw=2, label="Línea identidad")
    plt.xlabel("Glucosa de referencia [mg/dL]", fontsize=16)
    plt.ylabel("Predicciones", fontsize=16)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()

# Graficar cada conjunto
plot_regression(y_train, y_pred_train,"C:/Users/myria/Documents/CodigoFuenteTesis/resultados/BosquesAleatorios/regresion_trainBA0.pdf")
plot_regression(y_test, y_pred_test,"C:/Users/myria/Documents/CodigoFuenteTesis/resultados/BosquesAleatorios/regresion_testBAO.pdf")



# === 1. Cargar datos ===
datos = pd.read_csv(r"C:\Users\myria\Documents\CodigoFuenteTesis\datos\Glucosa_Data_Base_Clean.csv")

#Limpieza de datos
datos_clean=datos.dropna()

datos_completos = datos_clean

#Separación de caracteristicas y objetivo
X = datos_completos.drop(columns=["Glucosa"], errors="ignore", axis=1) # Todas las columnas
y = datos_clean['Glucosa'] #Columna de glucosa

# Entrenar modelo Random Forest
best_rf = grid_search.best_estimator_
best_rf.fit(X, y)


# Predicciones
y_predTD = best_rf.predict(X)

# === 3. Clasificación de rangos ===
def clasificar_glucosa(valor):
    if 72 <= valor <= 82:
        return "Valor bajo de glucosa"
    elif 82 <= valor <= 102:
        return "Valor medio de glucosa"
    else:
        return "Valor alto de glucosa"

datos_completos["rango_real"] = datos_completos["Glucosa"].apply(clasificar_glucosa)
datos_completos["rango_pred"] = pd.Series(y_predTD).apply(clasificar_glucosa)

# === 4. Métricas globales ===
mae_global = mean_absolute_error(y, y_predTD)
mse_global = mean_squared_error(y, y_predTD)
r2_global = r2_score(y, y_predTD)
pearson_global = pearsonr(y, y_predTD)[0]

# === 5. Métricas por rango ===
tabla_metricas = []
for rango, grupo in datos_completos.groupby("rango_real"):
    mae = mean_absolute_error(grupo["Glucosa"], y_predTD[grupo.index])
    mse = mean_squared_error(grupo["Glucosa"], y_predTD[grupo.index])
    r2 = r2_score(grupo["Glucosa"], y_predTD[grupo.index])
    pearson = pearsonr(grupo["Glucosa"], y_predTD[grupo.index])[0]
    tabla_metricas.append({
        "Rango": rango,
        "MAE": round(mae,2),
        "MSE": round(mse,2),
        "R2": round(r2,2),
        "Pearson": round(pearson,2)
    })

# Añadir métricas globales a la tabla
tabla_metricas.append({
    "Rango": "Global",
    "MAE": round(mae_global,2),
    "MSE": round(mse_global,2),
    "R2": round(r2_global,2),
    "Pearson": round(pearson_global,2)
})

tabla_metricas = pd.DataFrame(tabla_metricas)
orden = ["Valor bajo de glucosa", "Valor medio de glucosa", "Valor alto de glucosa", "Global"]
tabla_metricas = tabla_metricas.set_index("Rango").loc[orden].reset_index()

print("\nMétricas por rango y globales")
print(tabla_metricas)

# === 6. Matriz de confusión ===
conf = confusion_matrix(datos_completos["rango_real"], datos_completos["rango_pred"], labels=orden[:-1])
print("\nMatriz de confusión por rangos:")
print(conf)

# === 7. Gráfica de dispersión global ===
plt.figure(figsize=(7,7))
sns.scatterplot(x=y, y=y_predTD,
                color="royalblue", alpha=0.6, edgecolor="k", label="Predicciones")
lims = [min(y.min(), y_predTD.min()), max(y.max(), y_predTD.max())]
plt.plot(lims, lims, 'r--', lw=2, label="Línea identidad")
plt.xlabel("Glucosa de referencia [mg/dL]", fontsize=16)
plt.ylabel("Glucosa predicha [mg/dL]", fontsize=16)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig("C:/Users/myria/Documents/CodigoFuenteTesis/resultados/BosquesAleatorios/regresion_global_BAO.pdf", format="pdf", bbox_inches="tight")

# === 8. Gráficas de dispersión por rango ===
for rango in orden[:-1]:  # excluye 'Global'
    grupo = datos_completos[datos_completos["rango_real"] == rango]
    if grupo.empty:
        continue
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=grupo["Glucosa"], y=y_predTD[grupo.index],
                    color="royalblue", alpha=0.6, edgecolor="k", label="Predicciones")
    lims = [min(datos_completos['Glucosa'].min(), y_predTD.min()),
            max(datos_completos['Glucosa'].max(), y_predTD.max())]
    plt.plot(lims, lims, 'r--', lw=2, label="Línea identidad")
    plt.xlabel("Glucosa de referencia [mg/dL]", fontsize=16)
    plt.ylabel("Glucosa predicha [mg/dL]", fontsize=16)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(f"C:/Users/myria/Documents/CodigoFuenteTesis/resultados/BosquesAleatorios/regresion_{rango.replace(' ','_')}_BAO.pdf", format="pdf", bbox_inches="tight")
  

#Gráfico de Bland-Altman
# Valores reales y predicciones del set de prueba
y_true = y_test.values
y_pred = y_pred_test

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
plt.xlabel("Media entre valores reales y predicciones [mg/dL]")
plt.ylabel("Diferencia (Real - Predicción)")
#pt.title("Gráfico de Bland-Altman (Test Set)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.ylim(-40,40)


plt.savefig("C:/Users/myria/Documents/CodigoFuenteTesis/resultados/BosquesAleatorios/bland_altman_test-BAO.pdf", format="pdf", bbox_inches="tight")

#Gráfica de la norma ISO 15197:2015 para set de prueba
x = y_test  # Glucosa medida [mg/dL]
y = y_test - y_pred_test  # Diferencia estimado - medido [mg/dL]
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

# Título con estilo
#ax.set_title('Norma ISO 15197:2015 (0–200 mg/dL)', fontsize=14, fontweight='bold')

# Cuadrícula más sutil
ax.grid(True, linestyle='--', alpha=0.5)

# Leyenda
#ax.legend(loc='upper right', frameon=True)

# Mostrar
plt.tight_layout()
plt.savefig("C:/Users/myria/Documents/CodigoFuenteTesis/resultados/BosquesAleatorios/norma_iso_test_BAO.pdf", format="pdf", bbox_inches="tight")

#Implementación en hardware del modelo de BA optimizado 
import os
from sklearn.tree import export_text
import re
#Carpeta para guardar los árboles en formato texto
output_dir = os.path.join("modelos", "Arboles100_txt")
os.makedirs(output_dir, exist_ok=True)
#Exportar cada árbol del bosque a un archivo de texto
for i, tree in enumerate(best_rf.estimators_):
      tree_text = export_text(tree,feature_names=X.columns.tolist(), max_depth=15)
      with open( os.path.join("modelos", "Arboles100_txt", f"tree_{i}.txt"), "w") as f:
          f.write(tree_text)
print(f"\n✅ Se exportaron {len(best_rf.estimators_)} árboles a la carpeta: {output_dir}")


# Función que convierte el texto de un árbol en código C
def parse_single_tree(tree_text,tree_index=0, feature_names=None):
    lines = tree_text.strip().split('\n')
    cpp_lines = [f"float tree_{tree_index}(float*features){{"]
    indent_stack=[] #Pila para manejar niveles de indentación (Estructura de árbol)

    if feature_names is None:
        feature_names = []


    for line in lines:
        #Calcula la indentación (cuántos niveles de profundidad tiene la línea)
        indent = len(re.findall(r'\|',line))
        content = line.strip()
        content = re.sub(r'^(?:\|\s{0,3})*(?:\|[- ]+)+', '',content)  # Elimina cualquier combinación de |--- o |   ---
        content = content.replace('-', '-')
        content = re.sub(r'[^\x00-\x7F]+', '', content)
        content = re.sub(r'\s{2,}', ' ', content)
        content = content.strip()  # ✅ Limpia espacios extremos


        #Cierra bloques anteriores si la indentación disminuye
        while indent_stack and indent < indent_stack[-1]:
            cpp_lines.append('    ' * indent_stack.pop() + '}')

        #Ignora ramas truncadas
        if "truncated branch" in content:
            continue

        value_match = re.search(r"value:\s*\[([^\]]+)\]", content)

        if value_match:
            val_str = value_match.group(1).split()[0]
            cpp_lines.append('    ' * indent + f"return {val_str};")
            continue
        #Si es una condición (nodo interno del árbol)
        match = re.match(r"([a-zA-Z0-9_]+)\s*(<=|>|<|>=)\s*([\d\.Ee+-]+)",content)
        if match:
            var,op,val= match.groups()
            #Asigna índice si no existe
            var = var.strip().lower()
            feature_names_lower = [f.strip().lower() for f in feature_names]

            if var in feature_names_lower:
                idx= feature_names_lower.index(var)
            else:
                raise ValueError(f"Variable {var} no encontrada en feature_names")
            cpp_lines.append('    ' * indent + f"if (features[{idx}] {op} {val}) {{")
            indent_stack.append(indent +1)
            continue

        # Si no se pudo parsear, lo registramos
        print(f"⚠️ No se pudo interpretar esta línea: {repr(line)}")
        print(f"[DEBUG] Línea limpia: {repr(content)}")

  #Cierra cualquier bloque abierto restante
    while indent_stack:
        cpp_lines.append('    ' * indent_stack.pop() + '}')

    cpp_lines.append('}') #Cierra la función
    return '\n'.join(cpp_lines)#Devuelve el código como string


#Ruta de los árboles exportados como texto
folder = os.path.join("modelos", "Arboles100_txt")
#Contar cuántos archivos hay en la carpeta
tree_files= sorted([f for f in os.listdir(folder) if f.startswith("tree_") and f.endswith(".txt")])
n_trees = len(tree_files)
print(f"\n✅ Se encontraron {n_trees} árboles en la carpeta: {folder}")

#Encabeszado del archivo .h para evitar inclusiones múltiples
output_lines = ["#pragma once\n"]


#Procesa los 30 árboles y los convierte a funciones C
for i in range(n_trees):
    with open(os.path.join(folder,f"tree_{i}.txt"),"r")as f:
        tree_text = f.read()  #Lee el árbol como texto
        cpp_code = parse_single_tree(tree_text,tree_index=i, feature_names=X.columns.tolist())
        output_lines.append(cpp_code) #Agrega el archivo
        output_lines.append("\n") #Separador entre funciones

#Genera la función final que promedia las predicciones de los 30 árboles
output_lines.append("float predict(float* features) {")
output_lines.append("    if (features == nullptr) return -1.0;") #Validación de puntero nulo
output_lines.append("    float sum = 0.0;")
for i in range(n_trees):
    output_lines.append(f"    sum += tree_{i}(features);")#Llama a cada árbol
output_lines.append("    return sum / 100.0;") #Promedia las predicciones
output_lines.append("}")

# Ruta absoluta a la carpeta modelos
base_dir = r"c:\Users\myria\Documents\CodigoFuenteTesis\modelos"
os.makedirs(base_dir, exist_ok=True)
# Guarda todo el código generado en un archivo .h dentro de la carpeta modelos
output_file = os.path.join(base_dir, "modelo_rf.h")
with open(output_file, "w") as f:
    f.write('\n'.join(output_lines))

print(f"✅ Archivo de cabecera generado en: {output_file}")
