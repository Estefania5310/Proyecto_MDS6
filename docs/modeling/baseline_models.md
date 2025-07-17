

# Reporte del Modelo Baseline

## Descripción del modelo

Para iniciar la exploración de modelos, partimos del Modelo Cero. Este enfoque es crucial porque nos brinda 
una métrica de referencia clara en términos de precisión y pérdida. Al asegurar que este modelo base no presenta 
sobreajuste ni infraajuste evidentes, podemos evaluar con confianza si las modificaciones subsiguientes, 
como la incorporación de nuevas capas, el ajuste de hiperparámetros o la aplicación de otras técnicas, resultan en mejoras verdaderamente significativas.

## Variables de entrada

Las **variables de entrada** para el modelo *baseline* consisten en **lotes (batches) de imágenes de hojas**, 
las cuales son cargadas y preprocesadas directamente utilizando la función `keras.utils.image_dataset_from_directory`. 
Cada lote se estructura como un **tensor NumPy** con dimensiones de:
```math
(batch\_size, img\_height, img\_width, num\_channels),  
lo que se traduce específicamente en `$(128, 180, 180, 3)$`. Esto significa que el tamaño del lote (`batch_size`) es de $128$ imágenes, 
y cada imagen tiene unas dimensiones estandarizadas de ${180 \times 180}$ píxeles con $3$ canales para las imágenes RGB. Es crucial destacar que, 
previo a la alimentación al modelo, el conjunto de datos de entrenamiento fue **ampliado significativamente mediante técnicas de *Data Augmentation***. 
Esto permitió generar variaciones sintéticas de las imágenes existentes, incrementando la diversidad y cantidad de los datos de entrada, 
lo cual es fundamental para mejorar la robustez y la capacidad de generalización del modelo en tareas de clasificación de imágenes.

---

## Variable Objetivo del Modelo *Baseline*

La **variable objetivo** para el modelo *baseline* es la **categoría de salud de la hoja**, la cual es una variable categórica discreta con tres clases mutuamente excluyentes: **'Healthy'**, **'Powdery'** y **'Rust'**. Para el entrenamiento del modelo, estas clases se representan internamente mediante una **codificación *one-hot***. Esto significa que cada categoría se convierte en un vector binario de tres dimensiones (por ejemplo, `$[1, 0, 0]$` para 'Healthy', `$[0, 1, 0]$` para 'Powdery' y `$[0, 0, 1]$` para 'Rust'). El objetivo del modelo es predecir con precisión una de estas tres categorías para cada imagen de hoja de entrada.

---

## Evaluación del modelo

### Métricas de evaluación
Para evaluar el rendimiento del modelo *baseline*, se utilizaron dos métricas clave:

---

#### 1. Función de Pérdida (Loss Function)

La función de pérdida, representada por la variable **`loss`**, cuantifica la discrepancia entre las predicciones del modelo y las etiquetas verdaderas. Es un indicador de qué tan bien el modelo está aprendiendo y ajustándose a los datos.

* **Definición Técnica:** En problemas de clasificación multiclase como este, la métrica de pérdida comúnmente utilizada es la **entropía cruzada categórica (Categorical Cross-Entropy)** o **sparse categorical cross-entropy** (dependiendo de si las etiquetas están en formato one-hot o entero, respectivamente). Esta función penaliza al modelo por cada predicción incorrecta y por la confianza que tiene en sus predicciones erróneas.
* **Interpretación:** Un **valor de `loss` más bajo** indica que el modelo está realizando predicciones más precisas y confiables, lo que sugiere un mejor ajuste a los datos de prueba.

---

#### 2. Precisión (Accuracy)

La precisión, representada por la variable **`accuracy`**, es la métrica más intuitiva y ampliamente utilizada para problemas de clasificación. Mide la proporción de predicciones correctas del modelo.

* **Definición Técnica:** La precisión se calcula como el **número de predicciones correctas dividido por el número total de muestras** en el conjunto de datos de prueba.
    $$\text{Accuracy} = \frac{\text{Número de Predicciones Correctas}}{\text{Número Total de Muestras}}$$
* **Interpretación:** Un **valor de `accuracy` más alto** indica que el modelo clasifica correctamente una mayor proporción de las imágenes en el conjunto de prueba. Por ejemplo, una precisión del $0.90$ (o $90\%$) significa que el modelo clasificó correctamente el $90\%$ de las imágenes.

---

## Resultados de evaluación

El modelo *baseline* fue entrenado y evaluado, obteniendo los siguientes resultados.

### Arquitectura del Modelo

La arquitectura del modelo, denominada "sequential", está compuesta por capas convolucionales (`Conv2D`), capas de *pooling* (`MaxPooling2D`), una capa de aplanamiento (`Flatten`) y una capa densa final para la clasificación. La configuración específica de las capas se detalla a continuación:

```

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param \# ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 178, 178, 32)   │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max\_pooling2d (MaxPooling2D)    │ (None, 89, 89, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d\_1 (Conv2D)               │ (None, 87, 87, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max\_pooling2d\_1 (MaxPooling2D)  │ (None, 43, 43, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d\_2 (Conv2D)               │ (None, 41, 41, 128)    │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max\_pooling2d\_2 (MaxPooling2D)  │ (None, 20, 20, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 51200)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 3)              │       153,603 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
Total params: 246,851 (964.26 KB)
Trainable params: 246,851 (964.26 KB)
Non-trainable params: 0 (0.00 B)

```

### Proceso de Entrenamiento

El entrenamiento del modelo se realizó durante $15$ épocas. A continuación, se presentan las métricas de pérdida (`loss`) y precisión (`accuracy`) tanto para el conjunto de entrenamiento como para el conjunto de validación (`val_loss`, `val_accuracy`) a lo largo de las primeras épocas:

```

\--- Entrenando el modelo ---
Epoch 1/15 10/10 ━━━━━━━━━━━━━━━━━━━━ 620s 61s/step - accuracy: 0.3565 - loss: 1.1807 - val\_accuracy: 0.4333 - val\_loss: 1.0345
Epoch 2/15 10/10 ━━━━━━━━━━━━━━━━━━━━ 5s 428ms/step - accuracy: 0.5000 - loss: 1.0507 - val\_accuracy: 0.6500 - val\_loss: 1.0040
Epoch 3/15 10/10 ━━━━━━━━━━━━━━━━━━━━ 143s 8s/step - accuracy: 0.5255 - loss: 0.9657 - val\_accuracy: 0.5167 - val\_loss: 0.8616
Epoch 4/15 10/10 ━━━━━━━━━━━━━━━━━━━━ 9s 428ms/step - accuracy: 0.5469 - loss: 0.8223 - val\_accuracy: 0.5167 - val\_loss: 0.7723
Epoch 5/15 10/10 ━━━━━━━━━━━━━━━━━━━━ 82s 8s/step - accuracy: 0.6392 - loss: 0.7674 - val\_accuracy: 0.6833 - val\_loss: 0.6622
Epoch 6/15 10/10 ━━━━━━━━━━━━━━━━━━━━ 9s 411ms/step - accuracy: 0.7656 - loss: 0.6034 - val\_accuracy: 0.7500 - val\_loss: 0.5580
Epoch 7/15 10/10 ━━━━━━━━━━━━━━━━━━━━ 83s 8s/step - accuracy: 0.6745 - loss: 0.7094 - val\_accuracy: 0.7833 - val\_loss: 0.6202
Epoch 8/15 10/10 ━━━━━━━━━━━━━━━━━━━━ 8s 349ms/step - accuracy: 0.8203 - loss: 0.5560 - val\_accuracy: 0.8000 - val\_loss: 0.5839
Epoch 9/15 10/10 ━━━━━━━━━━━━━━━━━━━━ 79s 8s/step - accuracy: 0.7435 - loss: 0.5893 - val\_accuracy: 0.8333 - val\_loss: 0.4772
Epoch 10/15 10/10 ━━━━━━━━━━━━━━━━━━━━ 9s 451ms/step - accuracy: 0.7188 - loss: 0.6578 - val\_accuracy: 0.7833 - val\_loss: 0.5242
Epoch 11/15 10/10 ━━━━━━━━━━━━━━━━━━━━ 78s 8s/step - accuracy: 0.7623 - loss: 0.5534 - val\_accuracy: 0.8167 - val\_loss: 0.4537
Epoch 12/15 10/10 ━━━━━━━━━━━━━━━━━━━━ 7s 353ms/step - accuracy: 0.8359 - loss: 0.4461 - val\_accuracy: 0.8167 - val\_loss: 0.4663
Epoch 13/15 10/10 ━━━━━━━━━━━━━━━━━━━━ 78s 8s/step - accuracy: 0.8369 - loss: 0.4239 - val\_accuracy: 0.7833 - val\_loss: 0.5524
Epoch 14/15 10/10 ━━━━━━━━━━━━━━━━━━━━ 9s 402ms/step - accuracy: 0.8516 - loss: 0.3921 - val\_accuracy: 0.8167 - val\_loss: 0.3962
Epoch 15/15 10/10 ━━━━━━━━━━━━━━━━━━━━ 141s 9s/step - accuracy: 0.8589 - loss: 0.3890 - val\_accuracy: 0.8500 - val\_loss: 0.4032

```

### Rendimiento en el Conjunto de Prueba

Finalmente, el modelo fue evaluado en un conjunto de datos de prueba independiente para estimar su capacidad de generalización. Los resultados obtenidos fueron:

```

\--- Evaluando el modelo en el conjunto de prueba ---
2/2 ━━━━━━━━━━━━━━━━━━━━ 10s 1s/step - accuracy: 0.8245 - loss: 0.6704

```

* **Pérdida en el conjunto de prueba:** $0.6460$
* **Precisión en el conjunto de prueba:** $0.8267$

Estos resultados indican que el modelo *baseline* logró una **precisión superior al $82\%$** en la clasificación de las imágenes de hojas en el conjunto de datos no visto, con un valor de pérdida razonable, lo que sugiere un buen desempeño para el punto de partida.
```

------------------------------------------------------------------------------------------------------------


### Resultados de evaluación

Tabla que muestra los resultados de evaluación del modelo baseline, incluyendo las métricas de evaluación.

## Análisis de los resultados

Descripción de los resultados del modelo baseline, incluyendo fortalezas y debilidades del modelo.

## Conclusiones

Conclusiones generales sobre el rendimiento del modelo baseline y posibles áreas de mejora.

## Referencias

Lista de referencias utilizadas para construir el modelo baseline y evaluar su rendimiento.

Espero que te sea útil esta plantilla. Recuerda que puedes adaptarla a las necesidades específicas de tu proyecto.
