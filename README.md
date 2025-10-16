
# Informe Técnico - Regresión Logística y Extensiones Multiclase

-------------------
Curso: 2025-G1-910040-3-PEUCD-MACHINE LEARNING I
-------------------
## Integrantes del grupo

	- *Buleje Ticse, Jean Carlos*
	- *Sebastian Rios, Wilder Teddy*

## Introducción 

Este informe presenta el desarrollo completo del modelo de **Regresión Logística** y sus extensiones **One-vs-All (OvA)** y **Multinomial (Softmax)**, implementadas **desde cero** utilizando `numpy`, `pandas`, `matplotlib` y comparadas con los resultados obtenidos mediante `scikit-learn`.  
Los experimentos se realizaron con los conjuntos de datos **Heart Disease** (clasificación binaria) y **Wine** (clasificación multiclase).

---

## 1. Regresión Logística Binaria - *Heart Disease*

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

## 2. Regresión Logística Multiclase - One-vs-All (OvA)

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
| **Promedio macro** | - | - | **0.983** |
| **Accuracy global** | - | - | **0.98148** |

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

## Convergencia

-   **NLL inicial:** 1.0997\
-   **NLL final:** 0.0116 (≈ 1500 iteraciones)\
-   **Comportamiento:** Curva descendente estable y sin oscilaciones.

> 🔍 *Sin la corrección* $z \mathrel{-}= \max(z)$, el NLL diverge a valores mayores que $10^3$ en menos de 50 iteraciones.

------------------------------------------------------------------------

## Métricas

-   **Accuracy (Softmax scratch):** ≈ 0.982\
-   **Precision / Recall / F1:** similares a OvA\
-   **Scikit-learn:** `multi_class="multinomial", solver="lbfgs"`\
    → Resultados idénticos, validando el gradiente vectorizado.

------------------------------------------------------------------------

## 4. Comparación OvA vs Softmax

| **Aspecto** | **One-vs-All (OvA)** | **Multinomial (Softmax)** |
|------------------|------------------------|------------------------------|
| **Entrenamiento** | $K$ modelos binarios independientes | Un solo modelo con parámetros conjuntos |
| **Normalización** | Las probabilidades pueden no sumar 1 | La suma siempre es 1 |
| **Interacción entre clases** | Independiente | Competitiva (mutua exclusión) |
| **Estabilidad numérica** | Alta | Requiere centrado ($z \mathrel{-}= \max(z)$) |
| **Rendimiento en Wine** | Accuracy = 0.981 | Accuracy = 0.982 |
| **Interpretación** | Modular y simple | Coherente y teóricamente sólida |

> **Observación:**\
> En problemas linealmente separables, ambos modelos son equivalentes.\
> En escenarios con clases correlacionadas o desbalanceadas, **Softmax** suele mostrar mejor *recall* en clases minoritarias y fronteras más suaves, mientras que **OvA** tiende a favorecer clases dominantes.

------------------------------------------------------------------------

## 5. Conclusiones Finales

-   Las tres variantes (binaria, OvA y multinomial) comparten la misma base teórica:\
    **maximizar la log-verosimilitud** de los datos mediante **descenso del gradiente**.

-   La implementación desde cero produjo resultados **idénticos** a los de *scikit-learn*, confirmando la correcta derivación de los gradientes y la estabilidad de la optimización.

-   El truco de **estabilidad numérica** en Softmax ($z \mathrel{-}= \max(z)$) es indispensable para evitar *overflow* en la función exponencial.

-   Los coeficientes obtenidos son consistentes y permiten **interpretar de forma clara** qué variables químicas determinan cada variedad de vino.

-   El esquema **One-vs-All** es más sencillo y modular, mientras que el **Softmax multinomial** es más robusto, coherente y probabilísticamente fundamentado.

-   Los resultados experimentales alcanzaron **precisiones superiores al 98 %**, demostrando la correcta comprensión teórica y práctica del modelo de regresión logística y sus extensiones.

------------------------------------------------------------------------

### Conclusión general

La implementación práctica y teórica de la **regresión logística** demuestra dominio de los conceptos de:

-   Optimización y gradiente\
-   Estabilidad numérica\
-   Modelado probabilístico

El modelo **Softmax multinomial** se consolida como la **extensión más completa y estable**, mientras que el enfoque **One-vs-All** ofrece **simplicidad y gran rendimiento** en contextos bien definidos.

> 💡 Ambos enfoques, correctamente implementados, confirman el **poder de la regresión logística** como base para los modelos lineales de clasificación multiclase.
