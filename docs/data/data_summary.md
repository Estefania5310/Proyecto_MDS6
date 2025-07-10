
# Reporte de Datos

## Resumen general de los datos

El conjunto de datos comprende un total de 1,532 imágenes distribuidas en tres segmentos o subcarpetas distintas. Todas las imágenes están en formato .jpg y el tamaño total del conjunto de datos es de 1290.55 MB.

Este conjunto de datos se clasifica en tres etiquetas principales: Healthy, Powdery y Rust. A continuación, se detallará la distribución de las imágenes por clase y se analizarán los tamaños de las imágenes para una comprensión más profunda de la composición y características del dataset.

## Resumen de calidad de los datos

Dado que nuestro conjunto de datos consta de 1532 imágenes, es crucial verificar su calidad inicial. Para ello, intentaremos abrir cada imagen y evaluar su integridad. Si una imagen resulta estar vacía o tiene todos sus píxeles con el mismo valor, la registraremos como un archivo problemático. Adicionalmente, este proceso nos permitirá identificar archivos corruptos o con formatos no reconocidos que no puedan abrirse correctamente.
Como resultado del análisis evaluado se concluye que ninguna de las imágenes presenta una mala calidad y se procede a desarrollar las siguientes etapas del proyecto con la totalidad de las imágenes.

El conjunto de datos no presenta problemas de desbalance entre las clases, pero el número limitado de imágenes es una consideración importante. Para mejorar significativamente el rendimiento de nuestro modelo durante el entrenamiento y ayudarlo a generalizar mejor, se realiza un aumento de datos (data augmentation). Esta técnica expandirá artificialmente el tamaño de nuestro dataset al generar nuevas imágenes a partir de las existentes mediante diversas transformaciones.

Si bien la mayoría de las imágenes en el conjunto de datos comparten tamaños similares, estas dimensiones son considerablemente grandes. Un tamaño de imagen tan grande puede ocasionar problemas de rendimiento durante la ejecución y el entrenamiento del modelo, consumiendo excesivos recursos de memoria y ralentizando el proceso.

Para mitigar estos inconvenientes y optimizar el flujo de trabajo, Se realiza un redimensionamiento de todas las imágenes. Esto permitirá mantener la calidad visual necesaria para el análisis, al mismo tiempo que reducimos la carga computacional para un entrenamiento más eficiente.


<img src="https://drive.google.com/uc?export=view&id=1YHLzrijTSwQDtV9O9HLOUkfX17zdT79W" width="80%">

--- Distribución de Tamaños de Imagen Originales ---
- Ancho promedio: 3986.06, Desviación estándar: 521.47
- Alto promedio: 2702.45, Desviación estándar: 385.13
- Ancho min: 2421, Ancho max: 5184
- Alto min: 1728, Alto max: 3456


## Variable objetivo

El conjunto de datos parece estar relativamente balanceado, con un número de imágenes comparable en las tres clases. No hay una clase significativamente subrepresentada o sobrerrepresentada, lo cual es sumamente deseable para el entrenamiento de modelos de aprendizaje automático, ya que ayuda a prevenir sesgos y mejora la capacidad de generalización del modelo.

<img src="https://drive.google.com/uc?export=view&id=1xjjEDieT7Uq2cyB5V-FFnFKMmvZcsm6G" width="80%">

| Clase                  | Conteo de Imágenes por Clase   |
|------------------------|--------------------------------|
| `Healthy`              | 528     |
| `Powdery Mildew`       | 504     |
| `Rust`                 | 500     |


## Relación entre variables explicativas y variable objetivo

Se realiza un análisis de la composición de color de las imágenes dentro de cada clase generando histogramas y calculando estadísticas para los canales Rojo, Verde y Azul (RGB), con esto se obtiene información valiosa sobre las características visuales que distinguen las plantas Healthy, Powdery y Rust.

Inicialmente se observa una muestra por imagen de cada clase.

<img src="https://drive.google.com/uc?export=view&id=15RHLixfGwJmOVXzku36bV3pJ4iuIMNbp" width="80%">

La imagen anterior ayuda a comprender las características iniciales de la clasificación de las clases, en términos de color se nota una diferencia de las clases powdery y rust con respecto a la clase healty, como objetivo el modelo a desarrollar debería aprender a diferenciar el color y la textura de la "capa" de mildiú polvoriento de las "manchas" de roya.

Estadísticas y gráficos de composicioones de color por clase:

<img src="https://drive.google.com/uc?export=view&id=17gNVaEz6sqLovMpRi17gFMAvL_BtshI3" width="80%">

Estadísticas para la Clase: **HEALTHY**

| Color        | Media     | Desviación Estándar  |
|--------------|-----------|----------------------|
| `Rojo`       | 0.439     | 0.159                |
| `Verde`      | 0.599     | 0.147                |
| `Azul`       | 0.405     | 0.176                |

<img src="https://drive.google.com/uc?export=view&id=1wvY576V8nSSTdCMKd4MZ9l6x77xIonTo" width="80%">

Estadísticas para la Clase: **POWDERY**

| Color        | Media     | Desviación Estándar  |
|--------------|-----------|----------------------|
| `Rojo`       | 0.484     | 0.183                |
| `Verde`      | 0.588     | 0.161                |
| `Azul`       | 0.441     | 0.194                |


<img src="https://drive.google.com/uc?export=view&id=1PFAkWfam_05q-r4tWPN9ssyQk8KJtuJZ" width="80%">

Estadísticas para la Clase: **RUST**

| Color        | Media     | Desviación Estándar  |
|--------------|-----------|----------------------|
| `Rojo`       | 0.440     | 0.197                |
| `Verde`      | 0.558     | 0.204                |
| `Azul`       | 0.394     | 0.222                |


**Healthy (Saludables)**: El verde es el color dominante (media de 0.599), indicando hojas vibrantes. Los canales rojo y azul tienen medias más bajas, lo que es coherente con una planta sana. La baja desviación estándar sugiere tonos de verde muy uniformes.

**Powdery (Mildiu polvoriento)**: Aunque el verde sigue siendo predominante (media de 0.588), observamos un aumento en las medias de los canales rojo (0.484) y azul (0.441). Esto sugiere la presencia de tonos más claros o blanquecinos debido al Mildiu polvoriento, que introduce una mezcla de estos colores sobre las hojas. Las desviaciones estándar son ligeramente más altas, indicando una mayor variabilidad tonal por la distribución irregular del polvo.

**Rust (Oxidadas)**: Aquí, el verde disminuye ligeramente (media de 0.558) y, crucialmente, las desviaciones estándar para todos los canales son las más altas (Rojo: 0.197, Verde: 0.204, Azul: 0.222). Esto indica una gran diversidad de colores y tonos en las imágenes. Es consistente con las manchas rojizas/marrones (altos valores en rojo y quizás verde para el marrón) que contrastan con áreas aún verdes y la variabilidad cromática de una enfermedad de este tipo.

