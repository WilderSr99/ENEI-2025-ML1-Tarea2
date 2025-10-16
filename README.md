# Informe Técnico — Regresión Logística y Extensiones Multiclase

Este informe presenta el desarrollo completo del modelo de **Regresión Logística** y sus extensiones **One-vs-All (OvA)** y **Multinomial (Softmax)**, implementadas **desde cero** utilizando Python, `numpy`, `pandas` y `matplotlib`, y comparadas con los resultados de `scikit-learn`.  
Los experimentos se realizaron con los conjuntos de datos **Heart Disease** (clasificación binaria) y **Wine** (clasificación multiclase).

---

## 1. Regresión Logística Binaria — *Heart Disease*

### Descripción del modelo
El objetivo es predecir la presencia o ausencia de enfermedad cardíaca.  
El modelo se basa en la función sigmoide:
\[
p = \sigma(w^\top x) = \frac{1}{1 + e^{-w^\top x}}
\]
y la **log-verosimilitud** se maximiza mediante:
\[
\nabla_w \ell = X^\top(y - p)
\]

### Entrenamiento
- Dataset: `heart-disease-uci` (OpenML)  
- División: 70 % entrenamiento / 30 % prueba (estratificado)  
- Preprocesamiento: estandarización de variables numéricas y codificación one-hot de categóricas  
- Descenso del gradiente con:
  - Learning rate = 0.01 y 0.05  
  - Máx. iteraciones = 1500  

### Resultados
- **Accuracy (modelo desde cero):** ≈ 0.85  
- **Precision:** 0.84 **Recall:** 0.83 **F1:** 0.83  
- **Scikit-learn (`solver="lbfgs"`):** Accuracy ≈ 0.86 F1 ≈ 0.85  
- Las curvas de *log-likelihood* mostraron convergencia monótona:  
  - η = 0.01 → convergencia lenta pero estable  
  - η = 0.05 → convergencia más rápida, leve oscilación inicial  

**Interpretación:** el gradiente descendente manual reproduce el comportamiento del optimizador `lbfgs`, validando la correcta derivación del gradiente.

---

## 2. Regresión Logística Multiclase — One-vs-All (OvA)

### Concepto
Cada clase se trata como un problema binario independiente (*clase k vs resto*):
\[
\nabla_{w_k} \ell = X^\top(y_k - p_k)
\]

### Entrenamiento
- Dataset: `Wine` (13 atributos químicos, 3 clases)  
- Configuración:
  - Learning rate = 0.05  
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

> `Scikit-learn (multi_class="ovr")` reprodujo idéntico rendimiento, confirmando la equivalencia teórica.

### Análisis de coeficientes

Las variables más influyentes para cada clase fueron:

| Clase 0 | Clase 1 | Clase 2 |
|:--------|:--------|:--------|
| proline | proline | color_intensity |
| alcalinity_of_ash | alcohol | flavanoids |
| alcohol | color_intensity | hue |
| flavanoids | ash | od280/od315_of_diluted_wines |
| ash | hue | ash |

- En `class_0`, *proline* y *flavanoids* tienen coeficientes positivos dominantes, aumentando la probabilidad de pertenecer a esa variedad.  
- En `class_1`, la combinación de *alcohol* y *color_intensity* es la más representativa.  
- En `class_2`, la variable *hue* es el principal diferenciador.

Los coeficientes del modelo manual y `sklearn` coincidieron con diferencias <10⁻⁴, evidenciando una implementación numéricamente precisa.

---

## 3. Regresión Logística Multinomial (Softmax)

### Concepto
El modelo Softmax extiende la regresión logística al caso multiclase mediante una normalización exponencial:

\[
p_k(x) = \frac{e^{\theta_k^\top x}}{\sum_j e^{\theta_j^\top x}}
\]

y el gradiente vectorizado:
\[
\nabla_\Theta \ell = X^\top (Y - P)
\]
donde \(Y\) es la codificación one-hot y \(P\) las probabilidades predichas.

### Entrenamiento
- Learning rate = 0.05  
- Iteraciones = 3000  
- Inicialización aleatoria controlada (`random_state`)  
- Ajuste de estabilidad numérica:  
  ```python
  z -= np.max(z, axis=1, keepdims=True)
  P = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
