

# Reporte del Modelo Baseline

## Descripción del modelo

Para iniciar la exploración de modelos, partimos del Modelo Cero. Este enfoque es crucial porque nos brinda 
una métrica de referencia clara en términos de precisión y pérdida. Al asegurar que este modelo base no presenta 
sobreajuste ni infraajuste evidentes, podemos evaluar con confianza si las modificaciones subsiguientes, 
como la incorporación de nuevas capas, el ajuste de hiperparámetros o la aplicación de otras técnicas, resultan en mejoras verdaderamente significativas.

## Variables de entrada

Las **variables de entrada** para el modelo *baseline* consisten en **lotes (batches) de imágenes de hojas**, 
las cuales son cargadas y preprocesadas directamente utilizando la función `keras.utils.image_dataset_from_directory`. 
Cada lote se estructura como un **tensor NumPy** con dimensiones de: $(`batch_size`, `img_height`, `img_width`, `num_channels`)$,  
lo que se traduce específicamente en $(128, 180, 180, 3)$. Esto significa que el tamaño del lote (`batch_size`) es de $128$ imágenes, 
y cada imagen tiene unas dimensiones estandarizadas de ${180 \times 180}$ píxeles con $3$ canales para las imágenes RGB. Es crucial destacar que, 
previo a la alimentación al modelo, el conjunto de datos de entrenamiento fue **ampliado significativamente mediante técnicas de *Data Augmentation***. 
Esto permitió generar variaciones sintéticas de las imágenes existentes, incrementando la diversidad y cantidad de los datos de entrada, 
lo cual es fundamental para mejorar la robustez y la capacidad de generalización del modelo en tareas de clasificación de imágenes.

---

## Variable Objetivo del Modelo *Baseline*

La **variable objetivo** para el modelo *baseline* es la **categoría de salud de la hoja**, la cual es una variable categórica discreta con tres 
clases mutuamente excluyentes: 
**'Healthy'**, **'Powdery'** y **'Rust'**. Para el entrenamiento del modelo, estas clases se representan internamente mediante una **codificación *one-hot***. 
Esto significa que cada categoría se convierte en un vector binario de tres dimensiones (por ejemplo, $[1, 0, 0]$ para 'Healthy', $[0, 1, 0]$ para 
'Powdery' y $[0, 0, 1]$ para 'Rust'). El objetivo del modelo es predecir con precisión una de estas tres categorías para cada imagen de hoja de entrada.

---

## Evaluación del modelo

### Métricas de evaluación
Para evaluar el rendimiento del modelo *baseline*, se utilizaron dos métricas clave:

---

#### 1. Función de Pérdida (Loss Function)

La función de pérdida, representada por la variable **`loss`**, cuantifica la discrepancia entre las predicciones del modelo y las etiquetas verdaderas. 
Es un indicador de qué tan bien el modelo está aprendiendo y ajustándose a los datos.

* **Definición Técnica:** En problemas de clasificación multiclase como este, la métrica de pérdida comúnmente utilizada es la **entropía cruzada categórica 
(Categorical Cross-Entropy)** o **sparse categorical cross-entropy** (dependiendo de si las etiquetas están en formato one-hot o entero, respectivamente). 
Esta función penaliza al modelo por cada predicción incorrecta y por la confianza que tiene en sus predicciones erróneas.
* **Interpretación:** Un **valor de `loss` más bajo** indica que el modelo está realizando predicciones más precisas y confiables, lo que sugiere un mejor 
ajuste a los datos de prueba.

---

#### 2. Precisión (Accuracy)

La precisión, representada por la variable **`accuracy`**, es la métrica más intuitiva y ampliamente utilizada para problemas de clasificación. 
Mide la proporción de predicciones correctas del modelo.

* **Definición Técnica:** La precisión se calcula como el **número de predicciones correctas dividido por el número total de muestras** en el conjunto 
de datos de prueba.

$$\text{Accuracy} = \frac{\text{Número de Predicciones Correctas}}{\text{Número Total de Muestras}}$$

* **Interpretación:** Un **valor de `accuracy` más alto** indica que el modelo clasifica correctamente una mayor proporción de las imágenes en el conjunto 
de prueba. Por ejemplo, una precisión del $0.90$ (o $90\%$) significa que el modelo clasificó correctamente el $90\%$ de las imágenes.

---

## Resultados de evaluación

El modelo *baseline* fue entrenado y evaluado, obteniendo los siguientes resultados.

### Arquitectura del Modelo

La arquitectura del modelo, denominada "sequential", está compuesta por capas convolucionales (`Conv2D`), capas de *pooling* (`MaxPooling2D`), 
una capa de aplanamiento (`Flatten`) y una capa densa final para la clasificación. La configuración específica de las capas se detalla a continuación:

Model: "sequential"
| Layer (type)       | Output Shape         | Param # |
| :----------------- | :------------------- | :------ |
| conv2d (Conv2D)    | (None, 178, 178, 32) | 896     |
| max_pooling2d (MaxPooling2D) | (None, 89, 89, 32)   | 0       |
| conv2d_1 (Conv2D)  | (None, 87, 87, 64)   | 18,496  |
| max_pooling2d_1 (MaxPooling2D) | (None, 43, 43, 64)   | 0       |
| conv2d_2 (Conv2D)  | (None, 41, 41, 128)  | 73,856  |
| max_pooling2d_2 (MaxPooling2D) | (None, 20, 20, 128)  | 0       |
| flatten (Flatten)  | (None, 51200)        | 0       |
| dense (Dense)      | (None, 3)            | 153,603 |

* Total params: 246,851 (964.26 KB)
* Trainable params: 246,851 (964.26 KB)
* Non-trainable params: 0 (0.00 B)

### Proceso de Entrenamiento

El entrenamiento del modelo se realizó durante $15$ épocas. A continuación, se presentan las métricas de pérdida (`loss`) y precisión (`accuracy`) 
tanto para el conjunto de entrenamiento como para el conjunto de validación (`val_loss`, `val_accuracy`) a lo largo de las primeras épocas:

| Época | Precisión (Train) | Pérdida (Train) | Precisión (Val) | Pérdida (Val) |
| :---- | :---------------- | :-------------- | :-------------- | :------------ |
| 1     | 0.3565            | 1.1807          | 0.4333          | 1.0345        |
| 2     | 0.5000            | 1.0507          | 0.6500          | 1.0040        |
| 3     | 0.5255            | 0.9657          | 0.5167          | 0.8616        |
| 4     | 0.5469            | 0.8223          | 0.5167          | 0.7723        |
| 5     | 0.6392            | 0.7674          | 0.6833          | 0.6622        |
| 6     | 0.7656            | 0.6034          | 0.7500          | 0.5580        |
| 7     | 0.6745            | 0.7094          | 0.7833          | 0.6202        |
| 8     | 0.8203            | 0.5560          | 0.8000          | 0.5839        |
| 9     | 0.7435            | 0.5893          | 0.8333          | 0.4772        |
| 10    | 0.7188            | 0.6578          | 0.7833          | 0.5242        |
| 11    | 0.7623            | 0.5534          | 0.8167          | 0.4537        |
| 12    | 0.8359            | 0.4461          | 0.8167          | 0.4663        |
| 13    | 0.8369            | 0.4239          | 0.7833          | 0.5524        |
| 14    | 0.8516            | 0.3921          | 0.8167          | 0.3962        |
| 15    | 0.8589            | 0.3890          | 0.8500          | 0.4032        |

### Rendimiento en el Conjunto de Prueba

Finalmente, el modelo fue evaluado en un conjunto de datos de prueba independiente para estimar su capacidad de generalización. Los resultados 
obtenidos fueron:

| Métrica              | Valor   |
| :------------------- | :------ |
| Precisión (Test)     | 0.8245  |
| Pérdida (Test)       | 0.6704  |

* **Pérdida en el conjunto de prueba:** $0.6460$
* **Precisión en el conjunto de prueba:** $0.8267$

------------------------------------------------------------------------------------------------------------

## Análisis de los resultados

veamos las pérdidas del modelo y la progresión del accuracy a lo largo de las épocas, en general se observa una brecha importante entre los resultados 
de entrenamiento versus los resultados de la base de test, tanto en la evolucón del accuracy como en la disminución del parámetro de pérdida.

![Gráfica de Precisión del Modelo Baseline](docs/modeling/grafica1.png)

Estos resultados indican que el modelo *baseline* logró una **precisión superior al $82\%$** en la clasificación de las imágenes de hojas en el 
conjunto de datos no visto, con un valor de pérdida razonable, lo que sugiere un buen desempeño para el punto de partida.

## Conclusiones

Conclusiones generales sobre el rendimiento del modelo baseline y posibles áreas de mejora.

## Referencias

Lista de referencias utilizadas para construir el modelo baseline y evaluar su rendimiento.

Espero que te sea útil esta plantilla. Recuerda que puedes adaptarla a las necesidades específicas de tu proyecto.
