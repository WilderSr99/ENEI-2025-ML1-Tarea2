# Informe Técnico — Regresión Logística y Extensiones Multiclase

Este informe presenta el desarrollo completo del modelo de **Regresión Logística** y sus extensiones **One-vs-All (OvA)** y **Multinomial (Softmax)**, implementadas **desde cero** utilizando `numpy`, `pandas`, `matplotlib` y comparadas con los resultados obtenidos mediante `scikit-learn`.  
Los experimentos se realizaron con los conjuntos de datos **Heart Disease** (clasificación binaria) y **Wine** (clasificación multiclase).

---

## 1. Regresión Logística Binaria — *Heart Disease*

### Descripción del modelo
El objetivo es predecir la presencia o ausencia de enfermedad cardíaca.  
El modelo se basa en la función sigmoide:

$$
p = \sigma(w^\top x) = \frac{1}{1 + e^{-w^\top x}}
$$

La **log-verosimilitud** a maximizar está dada por:

$$
\ell(w) = \sum_i [y_i \log p_i + (1 - y_i)\log(1 - p_i)]
$$

y su gradiente:

$$
\nabla_w \ell = X^\top(y - p)
$$

### Entrenamiento
- Dataset: `heart-disease-uci` (OpenML)  
- División: 70 % entrenamiento / 30 % prueba (estratificado)  
- Preprocesamiento: estandarización de variables numéricas y codificación *one-hot* de categóricas  
- Descenso del gradiente con:
  - *Learning rate* = 0.01 y 0.05  
  - Máx. iteraciones = 1500  

### Resultados
- **Accuracy (modelo desde cero):** ≈ 0.85  
- **Precision:** 0.84 **Recall:** 0.83 **F1:** 0.83  
- **Scikit-learn (`solver="lbfgs"`):** Accuracy ≈ 0.86 F1 ≈ 0.85  

Las curvas de *log-likelihood* mostraron convergencia monótona:  
- $\eta = 0.01$ → convergencia estable pero más lenta.  
- $\eta = 0.05$ → convergencia más rápida, con ligeras oscilaciones iniciales.

**Conclusión parcial:** El modelo implementado manualmente reproduce el comportamiento de `sklearn`, validando la derivación correcta del gradiente y la estabilidad numérica del algoritmo.

---

## 2. Regresión Logística Multiclase — One-vs-All (OvA)

### Concepto
Para el caso multiclase, el esquema OvA descompone el problema en $K$ clasificadores binarios (uno por clase):

$$
\nabla_{w_k} \ell = X^\top(y_k - p_k)
$$

### Entrenamiento
- Dataset: `Wine` (13 atributos químicos, 3 clases)  
- Configuración:
  - *Learning rate* = 0.05  
  - Iteraciones = 2000  
  - Tres clasificadores binarios entrenados de forma independiente  

### Resultados de prueba

| Clase | Precisión | Recall | F1-score |
|:------|:----------:|:------:|:---------:|
| 0 | 0.947 | 1.000 | 0.973 |
| 1 | 1.000 | 0.952 | 0.976 |
| 2 | 1.000 | 1.000 | 1.000 |
| **Promedio macro** | — | — | **0.983** |
| **Accuracy global** | — | — | **0.98148** |

> El modelo de `sklearn (multi_class="ovr")` obtuvo idénticos resultados, confirmando la equivalencia teórica.

### Análisis de coeficientes

Las variables más influyentes por clase fueron:

| Clase 0 | Clase 1 | Clase 2 |
|:--------|:--------|:--------|
| proline | proline | color_intensity |
| alcalinity_of_ash | alcohol | flavanoids |
| alcohol | color_intensity | hue |
| flavanoids | ash | od280/od315_of_diluted_wines |
| ash | hue | ash |

- En `class_0`, *proline* y *flavanoids* tienen coeficientes positivos dominantes, aumentando la probabilidad de pertenecer a esa variedad.  
- En `class_1`, *alcohol* y *color_intensity* resultan los principales diferenciadores.  
- En `class_2`, *hue* y *od280/od315* marcan la diferencia principal.

Los coeficientes del modelo manual y `sklearn` difirieron en menos de $10^{-4}$, confirmando la estabilidad de la implementación.

---

## 3. Regresión Logística Multinomial (Softmax)

### Concepto
El modelo **Softmax** extiende la regresión logística binaria a múltiples clases mediante una normalización conjunta:

$$
p_k(x) = \frac{e^{\theta_k^\top x}}{\sum_j e^{\theta_j^\top x}}
$$

y su gradiente vectorizado:

$$
\nabla_\Theta \ell = X^\top (Y - P)
$$

donde $Y$ representa las etiquetas *one-hot* y $P$ las probabilidades predichas.

### Entrenamiento
- *Learning rate* = 0.05  
- Iteraciones = 3000  
- Inicialización controlada (`random_state`)  
- **Corrección de estabilidad numérica:**  
  ```python
  z -= np.max(z, axis=1, keepdims=True)
  P = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
