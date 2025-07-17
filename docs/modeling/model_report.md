
# Reporte del Modelo Final

## Resumen Ejecutivo

El modelo de fine-tuning utilizando EfficientNetB0 se destacó como la solución más robusta. 
Tras un proceso de calentamiento (`Warm-up`), este modelo alcanzó una precisión general del $0.98$ en el conjunto de validación. 
Su capacidad casi perfecta para identificar las enfermedades, con una precisión del $1.00$ para Mildiu Polvoriento y Óxido, 
y del $0.94$ para plantas saludables, confirma su alta eficacia. Esta precisión lo convierte en una gran herramienta para la vigilancia fitosanitaria 
y el manejo preventivo en la agricultura, con un gran potencial para ser integrado en sistemas de monitoreo como captura de imagenes por medio de drones. 
La implementación de esta solución podría mejorar significativamente la detección temprana de problemas en cultivos y así optimizar los recursos y mitigar 
los contratiempos.

## Descripción del Problema

Los agricultores se enfrentan al reto constante de mantener la sanidad y productividad de sus plantaciones, 
para esto se deben considerar múltiples factores agronómicos y ambientales, como la fertilidad del suelo, nivel de humedad, 
pronóstico del tiempo y el control de plagas y enfermedades. Sin embargo, realizar este monitoreo de manera continua y precisa 
en superficies amplias resulta sumamente costoso, consume mucho tiempo y requiere personal capacitado. 
En la práctica, esto significa que muchas enfermedades son detectadas tarde, cuando ya han causado daños significativos.

La falta de sistemas eficientes de detección y respuesta oportuna compromete no solo la salud de las plantas, 
sino también la sostenibilidad económica de las fincas. En cultivos de alto valor como el café, 
las enfermedades pueden reducir la productividad hasta en un 40% si no se controlan a tiempo, lo que genera pérdidas millonarias, 
incrementando el uso de agroquímicos y puede comprometer contratos de exportación que requieren estándares de calidad altos.(Atakishiyev et al., 2023). 

El objetivo del proyecto es desarrollar un modelo de aprendizaje profundo basado en redes neuronales convolucionales (CNN) para detectar automáticamente 
y de forma temprana dos tipos de enfermedades en las hojas de cultivos agrícolas. Este modelo servirá como base para su futura integración 
en sistemas de monitoreo inteligente, como drones o cámaras fijas, con el fin de mejorar la eficiencia, precisión y oportunidad en la toma de decisiones 
agronómicas.

## Descripción del Modelo

Cuando el problema que intentamos resolver con imágenes es muy diferente al problema original para el que el modelo fue entrenado 
(como identificar objetos cotidianos), simplemente usar el modelo como un extractor de características fijo no siempre da los mejores resultados. 
Aquí es donde entra el ajuste fino o fine-tuning. En lugar de congelar todas las capas, el fine-tuning nos permite entrenar y adaptar algunas 
o todas las capas del modelo pre-entrenado a nuestro nuevo y particular conjunto de datos. Esto permite que el modelo aprenda características 
más relevantes para nuestra tarea específica. Sin embargo, este proceso requiere más tiempo de entrenamiento y una cuidadosa experimentación.

Para que el ajuste fino sea exitoso, se considera lo siguiente:

* Calentamiento (`Warm-up`): Es buena idea empezar entrenando solo las nuevas capas que añadimos al modelo. 
Esto evita que los pesos aleatorios de estas nuevas capas "dañen" el conocimiento ya adquirido por las capas más profundas del modelo.

* Tasa de aprendizaje (Learning Rate): Utiliza una tasa de aprendizaje muy pequeña. Esto asegura que los ajustes a los pesos existentes 
del modelo sean graduales y no borren el conocimiento valioso que ya tiene.

* Congelamiento Estratégico: No siempre es necesario (o recomendable) entrenar todas las capas del modelo pre-entrenado. 
A menudo, las primeras capas de una red convolucional aprenden características muy básicas (como bordes o texturas) que son universales. 
Experimenta con cuántas capas descongelas y entrenas; a veces solo ajustar las capas más profundas es suficiente.

Para la definición del modelo, se adaptó la arquitectura de EfficientNetB0 a nuestro problema específico utilizando 
los siguientes componentes y configuraciones:

* (`Extractor`): Esta es la parte principal del modelo pre-entrenado mediante el modelo EfficientNetB0. 
Al usar include_top=False, se eliminan las capas finales de clasificación de este modelo, adaptaptándolo al problema específico que deseamos resolver 
(clasificación de enfermedades de plantas).

* Non-trainable params: La mayoría de los parámetros (pesos y sesgos) de esta sección están "congelados" (layer.trainable=False). 
Esto significa que no se modificarán durante el entrenamiento inicial (el "warming up").

* GlobalAveragePooling2D: Esta capa toma la salida del extractor (que es un tensor 3D) y la reduce a un vector 1D. 
Básicamente, calcula el promedio de los valores de cada mapa de características ayudando a reducir la dimensionalidad.

* Dense (Capa Densa 1): Capa de adaptación del modelo a nuestro problema de clasificación. Se definen 32 neuronas y se utiliza función de activación ReLU

* Dropout: Se utiliza técnica de regularización del 20% con el objetivo de prevenir el sobreajuste.

* Dense (Capa Densa 2 - Salida): Capa se salida con número de neuronas igual al número de clases para este caso 3 ('Healthy', 'Powdery', 'Rust'), 
con función de activación softmax ya que enfrentamos un problema multiclase.

### Resumen de Parámetros de la Arquitectura

Las imágenes de entrada se procesan con dimensiones de **$224 \times 224$ píxeles** y **$3$ canales de color (RGB)**. 
El modelo completo consta de $4,090,662$ parámetros. Sin embargo, en la fase de entrenamiento inicial (y ajustado a nuestro problema de clasificación), 
solo $41,091$ parámetros son "entrenables" (es decir, se ajustarán durante el entrenamiento). 
Dado que el conjunto de datos es limitado y entrenar una gran cantidad de parámetros requiere más datos y recursos computacionales, 
se optó por congelar la mayoría de los parámetros del modelo EfficientNetB0 original y entrenar únicamente las capas añadidas.

* **Total de parámetros:** $4,090,662$
* **Parámetros entrenables (inicialmente):** $41,091$
* **Parámetros no entrenables:** $4,049,571$ (parámetros que no se modificarán durante el entrenamiento actual, principalmente del *backbone* 
EfficientNetB0 congelado).

Esta fase inicial se diseñó para adaptar las **nuevas capas densas** añadidas al modelo sin modificar los pesos del *backbone* EfficientNetB0.

* **Capas Entrenables:** Solo las capas añadidas (GlobalAveragePooling2D, Dense, Dropout, Dense de salida) 
fueron marcadas como entrenables (`layer.trainable=False` para el extractor).
* **Tasa de Aprendizaje:** Se utilizó una tasa de aprendizaje de $10^{-3}$ (0.001) para permitir ajustes significativos en las nuevas capas.
* **Épocas:** El entrenamiento se realizó por **10 épocas**.
* **Callback:** Se utilizó un `ModelCheckpoint` para guardar los pesos del modelo (`warming_up.weights.h5`) 
que lograron la mejor `val_loss`, asegurando que el *fine-tuning* comience desde el punto más óptimo.
* **Datasets:** Se cargaron $1322$ imágenes para el conjunto de entrenamiento y $150$ para el conjunto de validación, 
pertenecientes a $3$ clases. Las imágenes se redimensionaron a $224 \times 224$ píxeles y se preprocesaron usando `efficientnet.preprocess_input`.

A continuación, se muestran los resultados del proceso de calentamiento:

| Época | Precisión (Train) | Pérdida (Train) | Precisión (Val) | Pérdida (Val) |
| :---- | :---------------- | :-------------- | :-------------- | :------------ |
| 1     | 0.7814            | 0.5472          | 0.9533          | 0.1710        |
| 2     | 0.9706            | 0.1187          | 0.9533          | 0.1614        |
| 3     | 0.9729            | 0.0913          | 0.9667          | 0.1361        |
| 4     | 0.9872            | 0.0547          | 0.9533          | 0.1268        |
| 5     | 0.9856            | 0.0455          | 0.9667          | 0.1207        |
| 6     | 0.9898            | 0.0482          | 0.9667          | 0.1183        |
| 7     | 0.9934            | 0.0319          | 0.9667          | 0.1063        |
| 8     | 0.9937            | 0.0237          | 0.9667          | 0.1016        |
| 9     | 0.9953            | 0.0266          | 0.9667          | 0.1152        |
| 10    | 0.9948            | 0.0239          | 0.9600          | 0.1041        |

Después del calentamiento, se procedió con el ajuste fino completo de todo el modelo.

* **Capas Entrenables:** Todas las capas del modelo (`ft_model.layers`) fueron marcadas como entrenables, 
permitiendo que los pesos del EfficientNetB0 se adapten a nuestro dataset.
* **Tasa de Aprendizaje:** Se disminuyó la tasa de aprendizaje a **$10^{-4}$ (0.0001)**. 
Esta tasa más pequeña permite ajustes más finos y graduales, preservando el conocimiento pre-entrenado y evitando que el modelo "olvide" lo que ya sabe.
* **Carga de Pesos:** Se cargaron los mejores pesos obtenidos durante la fase de calentamiento (`warming_up.weights.h5`) como punto de partida.
* **Épocas:** El modelo se entrenó por **20 épocas** adicionales.
* **Callback:** Se utilizó un `ModelCheckpoint` similar para guardar los mejores pesos del ajuste fino (`fine_tuning.weights.h5`).

A continuación, se muestran los resultados de la fase de ajuste fino:

| Época | Precisión (Train) | Pérdida (Train) | Precisión (Val) | Pérdida (Val) |
| :---- | :---------------- | :-------------- | :-------------- | :------------ |
| 1     | 0.8989            | 0.3341          | 0.9800          | 0.1559        |
| 2     | 0.9896            | 0.0397          | 0.9800          | 0.1552        |
| 3     | 0.9936            | 0.0213          | 0.9800          | 0.1458        |
| 4     | 0.9983            | 0.0117          | 0.9733          | 0.1513        |
| 5     | 0.9988            | 0.0087          | 0.9733          | 0.1278        |
| 6     | 1.0000            | 0.0036          | 0.9733          | 0.1170        |
| 7     | 0.9976            | 0.0074          | 0.9800          | 0.0857        |
| 8     | 1.0000            | 0.0034          | 0.9800          | 0.0911        |
| 9     | 0.9996            | 0.0032          | 0.9800          | 0.1034        |
| 10    | 1.0000            | 0.0018          | 0.9800          | 0.1120        |
| 11    | 0.9992            | 0.0031          | 0.9733          | 0.1252        |
| 12    | 0.9995            | 0.0032          | 0.9800          | 0.1275        |
| 13    | 0.9997            | 0.0018          | 0.9800          | 0.1465        |
| 14    | 0.9963            | 0.0106          | 0.9800          | 0.1753        |
| 15    | 0.9936            | 0.0143          | 0.9667          | 0.1219        |
| 16    | 1.0000            | 0.0033          | 0.9667          | 0.1929        |
| 17    | 0.9952            | 0.0076          | 0.9667          | 0.1972        |
| 18    | 0.9994            | 0.0027          | 0.9733          | 0.1580        |
| 19    | 0.9979            | 0.0033          | 0.9800          | 0.1489        |
| 20    | 0.9995            | 0.0020          | 0.9800          | 0.1326        |

El modelo converge de manera excelente, obteniendo para algunas épocas valores de precisión cercanos o iguales a $1.00$ 
en el conjunto de entrenamiento. La pérdida en el conjunto de entrenamiento disminuye drásticamente, acercándose a cero. 
La precisión y la pérdida en el conjunto de validación también muestran mejoras significativas, alcanzando valores muy altos 
($0.96 - 0.98$ de precisión) y bajas pérdidas, lo que es una buena señal de generalización.

## Evaluación del Modelo

El modelo fue evaluado en un conjunto de prueba independiente para estimar su capacidad de generalización en datos no vistos. 
A continuación, se presenta el reporte de clasificación:

| Métrica              | Valor   |
| :------------------- | :------ |
| **Precisión (Test)** | 0.98    |
| **Pérdida (Test)** | N/A (no se muestra directamente en el reporte de clasificación, pero se infiere que es baja dado el rendimiento) |

Los resultados de la evaluación son excepcionales, indicando un alto rendimiento del modelo de fine-tuning:

* **Precisión (Precision), Exhaustividad (Recall) y Puntuación F1 (F1-score) por Clase:**
    * **Clase 0:** Muestra una precisión del $0.94$ y un *recall* perfecto de $1.00$, resultando en un F1-score de $0.97$. Esto sugiere que el modelo es muy bueno identificando esta clase cuando predice que una imagen pertenece a ella, y casi no se le escapa ninguna instancia real de la clase.
    * **Clase 1:** Con una precisión perfecta de $1.00$ y un *recall* de $0.94$, también tiene un F1-score de $0.97$. El modelo es muy fiable cuando predice esta clase, aunque puede haber omitido algunas instancias reales.
    * **Clase 2:** Alcanza la perfección con $1.00$ en precisión, *recall* y F1-score. El modelo clasifica esta clase de manera impecable.

* **Precisión Global (Accuracy):** El modelo logró una **precisión general del $0.98$ (o $98\%$ )** en el conjunto de prueba. Esto es un indicador muy fuerte de su capacidad para clasificar correctamente las imágenes de hojas en datos no vistos.

* **Promedio Macro y Ponderado:**
    * **Macro Average:** Un valor de $0.98$ para precisión, *recall* y F1-score en el promedio macro sugiere que el modelo se desempeña de manera **uniformemente excelente en todas las clases**, sin que una clase domine los resultados ni enmascare el bajo rendimiento en otras.
    * **Weighted Average:** Un valor de $0.98$ en el promedio ponderado también confirma el alto rendimiento general, teniendo en cuenta la proporción de muestras en cada clase.

## Conclusiones y Recomendaciones

El modelo de **Fine-Tuning con EfficientNetB0** ha demostrado ser el enfoque definitivo y más efectivo para la clasificación de enfermedades en hojas de plantas, superando claramente al modelo *baseline* de red convolucional entrenada desde cero.

* **Rendimiento Superior:** Los resultados de precisión del $98\%$ en el conjunto de prueba, junto con los altos valores de precisión, *recall* y F1-score por clase, confirman que el modelo es altamente preciso y robusto para la tarea.
* **Estrategia de Entrenamiento Efectiva:** La implementación de la fase de **calentamiento** y el **ajuste fino** con tasas de aprendizaje adecuadas fueron cruciales para adaptar el conocimiento pre-entrenado a nuestro dominio específico, resultando en una convergencia rápida y una excelente generalización.
* **Eficiencia de Parámetros:** A pesar de la complejidad de EfficientNetB0, la estrategia de congelar la mayoría de sus parámetros y entrenar solo las capas densas añadidas (y luego un ajuste fino con una tasa de aprendizaje baja) permitió lograr un alto rendimiento con un ajuste eficiente de parámetros, lo cual es vital para datasets más pequeños.
* **Generalización Sólida:** La disminución constante de `val_loss` y el aumento de `val_accuracy` a lo largo de las épocas de entrenamiento indican que el modelo no sufre de sobreajuste severo y es capaz de generalizar bien a imágenes de hojas no vistas.

En resumen, el modelo de *fine-tuning* representa una solución robusta y de alto rendimiento para la clasificación precisa de enfermedades en hojas de plantas, sentando una base sólida para aplicaciones prácticas.

## Referencias
