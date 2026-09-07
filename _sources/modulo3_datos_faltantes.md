# Módulo 3: Visualización de datos interactivos

## Tratamiento de datos faltantes y atípicos

Dataset: `ds_salaries.csv` — 607 filas, 8 columnas.

---

### 1. Carga del dataset

```python
import pandas as pd
import numpy as np

# Cargar el archivo CSV proporcionado
file_path = "ds_salaries.csv"
df = pd.read_csv(file_path)
```

```python
# Informacion del dataset
df.info()
```

**Salida:**

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 607 entries, 0 to 606
Data columns (total 8 columns):
 #   Column                Non-Null Count  Dtype
---  ------                --------------  -----
 0   Working_Year          595 non-null    float64
 1   Designation           600 non-null    object
 2   Experience            607 non-null    object
 3   Employment_Status     607 non-null    object
 4   Employee_Location     607 non-null    object
 5   Company_Size          607 non-null    object
 6   Remote_Working_Ratio  607 non-null    int64
 7   Salary_USD            597 non-null    float64
dtypes: float64(2), int64(1), object(5)
memory usage: 38.1+ KB
```

Solo tres variables tienen datos faltantes: `Working_Year`, `Designation` y `Salary_USD`.

---

### 2. Diagnóstico de datos faltantes

```python
# Copia del dataset
df_va = df.copy()

# Revisar los datos faltantes en el conjunto de datos
missing_values = df_va.isnull().sum()

# Calcular el porcentaje de valores faltantes
missing_percentage = (df_va.isnull().sum() / len(df_va)) * 100

# Crear un DataFrame con el análisis de datos faltantes
missing_data = pd.DataFrame({
    'Valores Faltantes': missing_values,
    'Porcentaje Faltante (%)': missing_percentage
})

missing_data
```

**Salida:**

| Variable | Valores Faltantes | Porcentaje Faltante (%) |
|---|---:|---:|
| Working_Year | 12 | 1.98 |
| Designation | 7 | 1.15 |
| Experience | 0 | 0.00 |
| Employment_Status | 0 | 0.00 |
| Employee_Location | 0 | 0.00 |
| Company_Size | 0 | 0.00 |
| Remote_Working_Ratio | 0 | 0.00 |
| Salary_USD | 10 | 1.65 |

---

## Eliminación de datos faltantes

> **Regla del curso:** para eliminar filas, el porcentaje de faltantes de la columna debe ser **menor al 5 %**. Ojo: lo que importa es la **suma** de faltantes por fila, porque al eliminar filas la estructura de los datos cambia y hay que verificar que las distribuciones no se alteren.

### 3. Umbral del 5 %

```python
# Coloquemos un umbral
umbral = len(df_va) * 0.05   # Por ejemplo, 5% del total de filas

# Selecciona columnas con valores faltantes <= umbral
cols_to_drop = df_va.columns[missing_values <= umbral]

# Elimina filas con valores faltantes en las columnas seleccionadas
df_va.dropna(subset=cols_to_drop, inplace=True)
```

`umbral = 607 * 0.05 = 30.35`. Las tres columnas con NaN (12, 7 y 10) están por debajo del umbral, así que **las 8 columnas** entran en `cols_to_drop`.

```python
# informacion del dataset ajustado
df_va.info()
```

Resultado: **578 filas** (se eliminaron 29 filas, un 4.78 % del total).

```python
# Revisar los datos faltantes en el conjunto de datos
missing_values = df_va.isnull().sum()

# Calcular el porcentaje de valores faltantes
missing_percentage = (df_va.isnull().sum() / len(df_va)) * 100

missing_data = pd.DataFrame({
    'Valores Faltantes': missing_values,
    'Porcentaje Faltante (%)': missing_percentage
})

missing_data
```

Todas las columnas quedan en 0 faltantes.

### 4. Otra forma de eliminar los datos faltantes

```python
# copia del dataset
df_2 = df.copy()

# eliminar las filas con Nans
df_clean = df_2.dropna()
```

```python
# Revisar los datos faltantes en el conjunto de datos
missing_values = df_clean.isnull().sum()

# Calcular el porcentaje de valores faltantes
missing_percentage = (df_clean.isnull().sum() / len(df_clean)) * 100

missing_data = pd.DataFrame({
    'Valores Faltantes': missing_values,
    'Porcentaje Faltante (%)': missing_percentage
})

missing_data
```

`df_clean` tiene **578 filas** y 0 faltantes: en este caso `dropna()` sin argumentos produce exactamente el mismo resultado que el procedimiento con umbral, porque todas las columnas estaban por debajo del 5 %.

---

## 5. ¿Cambió la estructura de los datos? (variables ya vistas en clase)

Al eliminar filas la estructura puede cambiar. Hay que comprobarlo con **pruebas estadísticas**, y la prueba depende del tipo de variable:

| Tipo de variable | Comparación gráfica | Prueba estadística |
|---|---|---|
| Categórica | Barras agrupadas de frecuencias | Chi-cuadrado de homogeneidad |
| Numérica | Curvas de densidad (KDE) | Kolmogorov–Smirnov de 2 muestras |

### 5.1 `Experience` (categórica)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Contar frecuencias en cada DataFrame
original_counts = df['Experience'].value_counts()
clean_counts = df_clean['Experience'].value_counts()

# Unir ambos conteos en un solo DataFrame
comparison = pd.concat(
    [original_counts.rename("Original"), clean_counts.rename("Limpio")],
    axis=1
).fillna(0)

# Crear gráfico de barras agrupado
comparison.plot(kind='bar', figsize=(10, 6))

plt.title("Comparación de valores en 'Experience' antes y después de eliminar NaN")
plt.xlabel("Nivel de experiencia")
plt.ylabel("Frecuencia")
plt.xticks(rotation=45)
plt.legend(title="Dataset")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
```

```python
from scipy.stats import chi2_contingency

# Alinear categorías para evitar errores
comparison_table = pd.concat([original_counts, clean_counts], axis=1,
                             keys=['Original', 'Limpio']).fillna(0)

# Prueba chi-cuadrado
chi2, p, dof, expected = chi2_contingency(comparison_table.T)

# Mostrar resultados
print("Prueba Chi-cuadrado de homogeneidad")
print("Estadístico chi²:", chi2)
print("Grados de libertad:", dof)
print("Valor p:", p)
```

**Salida:**

```text
Prueba Chi-cuadrado de homogeneidad
Estadístico chi²: 0.07153479107112373
Grados de libertad: 3
Valor p: 0.9950192620159025
```

Hipótesis de la prueba de homogeneidad: $H_0: p_1 = p_2 = p_3 = p_4$ (las proporciones de cada categoría son iguales antes y después). Con $p = 0.995 > 0.05$ **no se rechaza $H_0$**: las proporciones no cambian.

### 5.2 `Remote_Working_Ratio` (numérica)

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Ignorar warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Crear gráfico de densidad
plt.figure(figsize=(10, 6))
sns.kdeplot(df['Remote_Working_Ratio'].dropna(), label='Original', linewidth=2)
sns.kdeplot(df_clean['Remote_Working_Ratio'].dropna(), label='Limpio', linewidth=2, linestyle='--')

plt.title("Distribución de 'Remote_Working_Ratio' antes y después de eliminar NaN")
plt.xlabel("Remote Working Ratio")
plt.ylabel("Densidad")
plt.legend(title="Dataset")
plt.grid(alpha=0.3)
plt.show()
```

```python
# PRUEBA KS
from scipy.stats import ks_2samp

stat, p_value = ks_2samp(df['Remote_Working_Ratio'], df_clean['Remote_Working_Ratio'])

print("Prueba Kolmogorov-Smirnov (KS) para 'Remote_Working_Ratio' ")
print("Estadístico KS:", stat)
print("Valor p:", p_value)
if p_value < 0.05:
    print("✅ Hay diferencias significativas entre las dos distribuciones.")
else:
    print("❌ No hay evidencia significativa de que las distribuciones sean diferentes.")
```

**Salida:**

```text
Prueba Kolmogorov-Smirnov (KS) para 'Remote_Working_Ratio'
Estadístico KS: 0.0018469641951169458
Valor p: 1.0
❌ No hay evidencia significativa de que las distribuciones sean diferentes.
```

---

# TAREA: verificar el resto de las variables

Las variables `Experience` y `Remote_Working_Ratio` ya se revisaron en clase. Faltan las seis restantes:

| Variable | Tipo | Prueba usada |
|---|---|---|
| Working_Year | Numérica discreta (3 años) | Chi-cuadrado + KS |
| Designation | Categórica (50 categorías) | Chi-cuadrado |
| Employment_Status | Categórica (4 categorías) | Chi-cuadrado |
| Employee_Location | Categórica (57 categorías) | Chi-cuadrado |
| Company_Size | Categórica (3 categorías) | Chi-cuadrado |
| Salary_USD | Numérica continua | KS |

Para no repetir el mismo bloque seis veces se definen dos funciones reutilizables.

## Funciones de verificación

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ks_2samp
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def comparar_categorica(col, top=None):
    """Compara la distribución de una variable categórica antes/después de dropna().

    Grafica barras agrupadas y aplica la prueba chi-cuadrado de homogeneidad.
    `top` limita el gráfico a las categorías más frecuentes (solo visual).
    """
    original_counts = df[col].value_counts()
    clean_counts = df_clean[col].value_counts()

    # Alinear categorías para evitar errores: fillna(0) para las que desaparecen
    comparison = pd.concat([original_counts, clean_counts],
                           axis=1, keys=['Original', 'Limpio']).fillna(0)

    grafico = comparison.head(top) if top else comparison
    grafico.plot(kind='bar', figsize=(10, 6))
    plt.title(f"Comparación de valores en '{col}' antes y después de eliminar NaN")
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Dataset")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    chi2, p, dof, expected = chi2_contingency(comparison.T)
    print(f"Prueba Chi-cuadrado de homogeneidad para '{col}'")
    print("Estadístico chi²:", chi2)
    print("Grados de libertad:", dof)
    print("Valor p:", p)
    # El chi-cuadrado exige frecuencias esperadas suficientes; se reporta para juzgar validez
    print("Frecuencia esperada mínima:", round(expected.min(), 2))
    if p < 0.05:
        print("✅ Hay diferencias significativas entre las distribuciones.")
    else:
        print("❌ No hay evidencia significativa de que las distribuciones sean diferentes.")


def comparar_numerica(col):
    """Compara la distribución de una variable numérica antes/después de dropna().

    Grafica densidades superpuestas y aplica la prueba Kolmogorov-Smirnov.
    Se usa .dropna() en el original porque kdeplot y ks_2samp no admiten NaN.
    """
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df[col].dropna(), label='Original', linewidth=2)
    sns.kdeplot(df_clean[col], label='Limpio', linewidth=2, linestyle='--')
    plt.title(f"Distribución de '{col}' antes y después de eliminar NaN")
    plt.xlabel(col)
    plt.ylabel("Densidad")
    plt.legend(title="Dataset")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    stat, p_value = ks_2samp(df[col].dropna(), df_clean[col])
    print(f"Prueba Kolmogorov-Smirnov (KS) para '{col}'")
    print("Estadístico KS:", stat)
    print("Valor p:", p_value)
    print("Media   original / limpio:", round(df[col].mean(), 2), "/", round(df_clean[col].mean(), 2))
    print("Mediana original / limpio:", df[col].median(), "/", df_clean[col].median())
    print("Desv.   original / limpio:", round(df[col].std(), 2), "/", round(df_clean[col].std(), 2))
    if p_value < 0.05:
        print("✅ Hay diferencias significativas entre las dos distribuciones.")
    else:
        print("❌ No hay evidencia significativa de que las distribuciones sean diferentes.")
```

---

## A. `Working_Year`

Aunque es `float64`, solo toma tres valores (2020, 2021, 2022), así que se puede tratar como categórica ordinal. Se aplican ambas pruebas.

```python
comparar_categorica('Working_Year')
comparar_numerica('Working_Year')
```

![Working_Year barras](figuras/working_year.png)

![Working_Year densidad](figuras/working_year_kde.png)

**Salida:**

```text
Prueba Chi-cuadrado de homogeneidad para 'Working_Year'
Estadístico chi²: 0.14144865...
Grados de libertad: 2
Valor p: 0.9317186...
Frecuencia esperada mínima: 68.0
❌ No hay evidencia significativa de que las distribuciones sean diferentes.

Prueba Kolmogorov-Smirnov (KS) para 'Working_Year'
Estadístico KS: 0.006822...
Valor p: 1.0
Media   original / limpio: 2021.4 / 2021.42
Mediana original / limpio: 2022.0 / 2022.0
Desv.   original / limpio: 0.7 / 0.69
❌ No hay evidencia significativa de que las distribuciones sean diferentes.
```

| Año | Original (%) | Limpio (%) |
|---|---:|---:|
| 2022 | 52.44 | 53.11 |
| 2021 | 35.46 | 35.47 |
| 2020 | 12.10 | 11.42 |

**Conclusión:** la distribución **no cambia**. Es la variable con más faltantes (12) y aun así el corrimiento máximo es de 0.68 puntos porcentuales en 2022.

---

## B. `Designation`

```python
comparar_categorica('Designation', top=15)
```

![Designation](figuras/designation.png)

**Salida:**

```text
Prueba Chi-cuadrado de homogeneidad para 'Designation'
Estadístico chi²: 0.7188657...
Grados de libertad: 49
Valor p: 1.0
Frecuencia esperada mínima: 0.98
❌ No hay evidencia significativa de que las distribuciones sean diferentes.
```

| Designation | Original (%) | Limpio (%) |
|---|---:|---:|
| Data Scientist | 23.67 | 24.22 |
| Data Engineer | 21.83 | 21.45 |
| Data Analyst | 16.17 | 15.92 |
| Machine Learning Engineer | 6.50 | 6.57 |
| Research Scientist | 2.67 | 2.25 |

**Conclusión:** la distribución **no cambia**.

> **Advertencia estadística:** con 50 categorías, el 61 % de las frecuencias esperadas es menor a 5 y la mínima es 0.98. El supuesto del chi-cuadrado (esperadas ≥ 5) **no se cumple**, así que el valor p exacto no es confiable. La comparación de proporciones sí lo respalda: ninguna categoría se mueve más de 0.55 puntos porcentuales.

---

## C. `Employment_Status`

```python
comparar_categorica('Employment_Status')
```

![Employment_Status](figuras/employment_status.png)

**Salida:**

```text
Prueba Chi-cuadrado de homogeneidad para 'Employment_Status'
Estadístico chi²: 0.0258690...
Grados de libertad: 3
Valor p: 0.9989020...
Frecuencia esperada mínima: 3.9
❌ No hay evidencia significativa de que las distribuciones sean diferentes.
```

| Estado | Original (%) | Limpio (%) |
|---|---:|---:|
| FT (tiempo completo) | 96.87 | 96.89 |
| PT (medio tiempo) | 1.65 | 1.56 |
| CT (contrato) | 0.82 | 0.87 |
| FL (freelance) | 0.66 | 0.69 |

**Conclusión:** la distribución **no cambia**. La variable no tenía NaN propios, así que solo se ve afectada por las filas eliminadas por otras columnas. Nota: las categorías FL y CT tienen esperadas < 5, mismo caveat que arriba, pero con desviaciones de 0.09 pp o menos.

---

## D. `Employee_Location`

```python
comparar_categorica('Employee_Location', top=15)
```

![Employee_Location](figuras/employee_location.png)

**Salida:**

```text
Prueba Chi-cuadrado de homogeneidad para 'Employee_Location'
Estadístico chi²: 2.4554684...
Grados de libertad: 56
Valor p: 1.0
Frecuencia esperada mínima: 0.49
❌ No hay evidencia significativa de que las distribuciones sean diferentes.
```

| País | Original (%) | Limpio (%) |
|---|---:|---:|
| US | 54.70 | 55.19 |
| GB | 7.25 | 7.27 |
| IN | 4.94 | 5.19 |
| CA | 4.78 | 4.84 |
| DE | 4.12 | 3.81 |

**Conclusión:** la distribución **no cambia**. Mismo caveat que `Designation`: 57 categorías y 78 % de esperadas < 5, por lo que la evidencia fuerte aquí es la tabla de proporciones (máximo corrimiento: 0.49 pp en US), no el valor p.

---

## E. `Company_Size`

```python
comparar_categorica('Company_Size')
```

![Company_Size](figuras/company_size.png)

**Salida:**

```text
Prueba Chi-cuadrado de homogeneidad para 'Company_Size'
Estadístico chi²: 0.0118589...
Grados de libertad: 2
Valor p: 0.9940881...
Frecuencia esperada mínima: 78.53
❌ No hay evidencia significativa de que las distribuciones sean diferentes.
```

| Tamaño | Original (%) | Limpio (%) |
|---|---:|---:|
| M (mediana) | 53.71 | 53.98 |
| L (grande) | 32.62 | 32.53 |
| S (pequeña) | 13.67 | 13.49 |

**Conclusión:** la distribución **no cambia**. Aquí el chi-cuadrado sí es plenamente válido (esperada mínima 78.53).

---

## F. `Salary_USD`

```python
comparar_numerica('Salary_USD')
```

![Salary_USD](figuras/salary_usd.png)

**Salida:**

```text
Prueba Kolmogorov-Smirnov (KS) para 'Salary_USD'
Estadístico KS: 0.0068623...
Valor p: 1.0
Media   original / limpio: 106911.44 / 106989.75
Mediana original / limpio: 97489.0 / 98271.0
Desv.   original / limpio: 67087.27 / 67350.92
❌ No hay evidencia significativa de que las distribuciones sean diferentes.
```

**Conclusión:** la distribución **no cambia**. La media se mueve un 0.07 % y la mediana un 0.80 %; las dos curvas de densidad se superponen casi por completo, conservando la asimetría a la derecha típica de los salarios.

---

## Resumen de la verificación

| Variable | Tipo | Prueba | Estadístico | gl | Valor p | ¿Cambia la distribución? |
|---|---|---|---:|---:|---:|---|
| Experience *(clase)* | Categórica | Chi² | 0.0715 | 3 | 0.9950 | No |
| Remote_Working_Ratio *(clase)* | Numérica | KS | 0.0018 | — | 1.0000 | No |
| Working_Year | Num. discreta | Chi² / KS | 0.1414 / 0.0068 | 2 / — | 0.9317 / 1.0000 | No |
| Designation | Categórica | Chi² | 0.7189 | 49 | 1.0000 | No* |
| Employment_Status | Categórica | Chi² | 0.0259 | 3 | 0.9989 | No |
| Employee_Location | Categórica | Chi² | 2.4555 | 56 | 1.0000 | No* |
| Company_Size | Categórica | Chi² | 0.0119 | 2 | 0.9941 | No |
| Salary_USD | Numérica | KS | 0.0069 | — | 1.0000 | No |

\* Valor p poco confiable por frecuencias esperadas bajas (muchas categorías); la conclusión se sostiene en la comparación directa de proporciones.

### Conclusión general

En **ninguna** de las ocho variables se rechaza la hipótesis nula ($\alpha = 0.05$). Eliminar las 29 filas con datos faltantes (**4.78 %** del total, por debajo del umbral del 5 %) **no altera la estructura del conjunto de datos**: ni las proporciones de las variables categóricas ni las distribuciones de las numéricas.

Sin embargo, se debe señalar una limitación en dos casos. Designation tiene 50 categorías y Employee_Location tiene 57, y en ambas la mayoría de las frecuencias esperadas quedó por debajo de 5 (61 % y 78 % de las celdas respectivamente). Como el chi-cuadrado exige frecuencias esperadas suficientes, en estas dos variables el valor p no es confiable por sí solo. Por eso se comparo directamente las proporciones y comprobamos que ninguna categoría se desplaza más de 0.55 puntos porcentuales, que es la evidencia en la que realmentenos da para concluir que no hubo cambio.