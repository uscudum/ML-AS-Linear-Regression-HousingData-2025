# Introducción a la Regresión Lineal

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## Carga y visualización de datos con Pandas

Para trabajar con un archivo CSV en Python utilizando Pandas, seguimos tres pasos básicos:

1. Importamos la librería `pandas` con el alias `pd`.
2. Cargamos el archivo `HousingData.csv` utilizando la función `pd.read_csv()`.
3. Mostramos las primeras cinco filas del dataset con `df.head(5)`.

```python
import pandas as pd

df = pd.read_csv('HousingData.csv')
df.head(5)
```

Ejemplo de salida:

| CRIM   | ZN   | INDUS | CHAS | NOX  | RM   | AGE  | DIS   | RAD | TAX | PTRATIO | B     | LSTAT | MEDV |
|--------|------|-------|------|------|------|------|-------|-----|-----|---------|-------|-------|------|
| 0.0063 | 18.0 | 2.31  | 0.0  | 0.538| 6.575| 65.2 | 4.09  | 1   | 296 | 15.3    | 396.9 | 4.98  | 24.0 |
| 0.0273 | 0.0  | 7.07  | 0.0  | 0.469| 6.421| 78.9 | 4.9671| 2   | 242 | 17.8    | 396.9 | 9.14  | 21.6 |

---

## Visualización de relaciones entre variables con Seaborn

Generamos gráficos para analizar relaciones entre columnas del dataset.

```python
import seaborn as sns
import matplotlib.pyplot as plt

cols = ['CRIM','RM','AGE','MEDV']
sns.pairplot(df[cols])
plt.show()
```

---

## Analizando correlaciones entre variables

Podemos usar `.corr()` de Pandas y `heatmap` de Seaborn:

```python
sns.heatmap(df[cols].corr(), annot=True)
```

---

## Creación y ajuste de un modelo de Regresión Lineal con Scikit-learn

1. Definimos variables:
   - `X` = número medio de habitaciones por vivienda (`RM`).
   - `Y` = valor medio de las viviendas (`MEDV`).
2. Estandarizamos datos con `StandardScaler`.
3. Entrenamos el modelo con `LinearRegression`.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

X = df['RM'].values.reshape(-1,1)
Y = df['MEDV'].values.reshape(-1,1)

ss_x = StandardScaler()
ss_y = StandardScaler()

X_ss = ss_x.fit_transform(X)
Y_ss = ss_y.fit_transform(Y)

lr = LinearRegression()
lr.fit(X_ss, Y_ss)
```

---

## Predicciones

```python
import numpy as np

num_hab = 10
num_hab_std = ss_x.transform([[num_hab]])

result_std = lr.predict(num_hab_std)
result = ss_y.inverse_transform(result_std)

print(num_hab, "habitaciones se traducen a US$ ", round(result[0][0] * 1000 , 2), "aprox")
```

Ejemplo de salida:

```
10 habitaciones se traducen a US$  56350.47 aprox
```
