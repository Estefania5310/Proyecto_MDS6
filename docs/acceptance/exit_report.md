
# Informe de salida

## Resumen Ejecutivo

Este informe describe los resultados del proyecto de Machine Learning, destacando los logros principales y las lecciones aprendidas durante su ejecución. El objetivo fue desarrollar un modelo de aprendizaje profundo para la clasificación de datos de imagen, utilizando como caso de estudio la detección automática de enfermedades en hojas de plantas.

En la etapa de preprocesamiento de datos, se trabajó con un total de 1532 imágenes, donde se realizó la verificación de calidad de los datos, la normalización y redimensionamiento de las imágenes para estandarizar el formato de entrada, y la aplicación de técnicas de aumentación de datos (data augmentation). Esta última fue fundamental para expandir el conjunto de entrenamiento, lo que, a su vez, mejoró la robustez y la capacidad de generalización del modelo. Además, se realizó un análisis exploratorio para entender las características intrínsecas de los datos, como la distribución de clases y las propiedades visuales.

La fase de modelado implicó la exploración y desarrollo de arquitecturas de redes neuronales. Se inició con un modelo base (baseline) para establecer una referencia de rendimiento, logrando una precisión inicial del 82%. Sin embargo, el modelo final representó una mejora sustancial, al implementar una estrategia de transfer learning utilizando una arquitectura pre-entrenada como EfficientNetB0. Este modelo fue optimizado mediante un proceso de calentamiento (Warm-up) y un ajuste fino (fine-tuning) de sus capas, adaptando eficazmente su vasto conocimiento a nuestro dataset específico. Este enfoque resultó en un desempeño significativamente superior.

Finalmente, el despliegue del modelo se planificó y ejecutó para transformar el modelo entrenado en un servicio accesible. Si bien el desarrollo se realizó en un entorno de FastAPI en Google Colaboratory, se establecieron los requisitos técnicos detallados para una infraestructura de producción, incluyendo especificaciones de software y hardware. La contenerización con Docker fue identificada como un paso esencial para la portabilidad y consistencia del despliegue, y se diseñaron medidas de seguridad básicas, como la autenticación mediante claves API y el monitoreo de vulnerabilidades, para asegurar la fiabilidad y escalabilidad del servicio.

Los resultados generales del proyecto demuestran la capacidad de la metodología de Machine Learning para abordar problemas complejos de clasificación. El modelo final, con una precisión del 98%, no solo validó la efectividad de las técnicas aplicadas, sino que también sentó las bases para su futura integración en sistemas que requieran análisis predictivos de alta precisión.

## Resultados del proyecto

# Definición y Comprensión del Negocio:

El proyecto se centró en el desarrollo de un modelo de aprendizaje automático para la clasificación de datos visuales, con la aplicación específica en la detección de enfermedades en hojas de plantas. El objetivo principal fue construir un modelo de aprendizaje profundo (CNN) capaz de realizar clasificaciones precisas.

El alcance cubrió el ciclo de vida completo de un proyecto de Machine Learning, desde el procesamiento de datos hasta la evaluación del modelo y la preparación para el despliegue. Se adoptó la metodología TDSP (Team Data Science Process), asegurando un enfoque estructurado y colaborativo. El proyecto se ejecutó en un cronograma de 5 semanas.

Adquisición y Preprocesamiento de Datos:

Los datos se obtuvieron de un repositorio público de imágenes (Plant Disease Recognition Dataset de Kaggle). Se implementaron scripts para la carga y gestión de datasets de imágenes, incluyendo la organización por clases y la generación de conjuntos de entrenamiento, validación y prueba. Se realizó una evaluación exhaustiva de la calidad de los datos, confirmando la integridad de todas las imágenes (1,532 en formato .jpg, 1290.55 MB).

Se realizó un redimensionamiento de imágenes a tamaños estándar (180x180 para el baseline, 224x224 para el modelo final). Además, se usó una Aumentación de datos (rotaciones, brillo, zoom) para enriquecer el dataset y mejorar la capacidad de generalización del modelo.

Se llevó a cabo un análisis exploratorio de los datos, incluyendo la distribución de clases (balanceadas) y la composición de color, para comprender las características subyacentes relevantes para la clasificación.

#Modelado y Entrenamiento:

Como modelo Baseline Se desarrolló una red neuronal convolucional básica como punto de partida, con una arquitectura secuencial que incluía capas Conv2D, MaxPooling2D, Flatten y Dense. Se definió la variable de entrada como lotes de imágenes preprocesadas y la variable objetivo como la categoría de clase codificada (one-hot).

Las métricas de evaluación principales fueron la función de pérdida (Categorical Cross-Entropy) y la precisión (Accuracy).

Este modelo inicial alcanzó una precisión del 82% en el conjunto de prueba, sirviendo como una referencia sólida.

Se escogió cómo modelo final el modelo Transfer Learning con EfficientNetB0, donde el proceso de entrenamiento se dividió en dos fases clave:

- Fase de calentamiento (Warm-up): Entrenamiento inicial de las capas densas añadidas al backbone de EfficientNetB0 (10 épocas, tasa de aprendizaje 10−3 ), manteniendo la mayoría de los parámetros del modelo pre-entrenado congelados.

- Fase de ajuste fino (Fine-tuning): Descongelamiento y entrenamiento de todas las capas del modelo (20 épocas adicionales, tasa de aprendizaje reducida a 10−4) para adaptar con precisión el conocimiento pre-entrenado al dataset específico.

El modelo final demostró un rendimiento sobresaliente en el conjunto de prueba independiente. Precisión Global (Accuracy): Alcanzó un impresionante 0.98 (o 98% ). Este modelo superó drásticamente al Modelo Baseline, que había logrado una precisión del 82%. Esta mejora de 16 puntos porcentuales en la precisión demuestra la efectividad de la estrategia de transfer learning y fine-tuning para maximizar el rendimiento del modelo en tareas de clasificación de imágenes.

# Descripción de los resultados y su relevancia para el negocio

Los resultados obtenidos validan la eficacia de las técnicas de Machine Learning y aprendizaje profundo aplicadas a este tipo de problematicas.

Precisión y Robustez: La consecución de una precisión del 98% en la clasificación de imágenes indica que el modelo es altamente preciso y robusto, capaz de generar predicciones muy confiables en datos no vistos. Este nivel de rendimiento es crucial para la toma de decisiones basada en datos en cualquier aplicación empresarial.

Eficiencia en el Desarrollo: La estrategia de transfer learning permitió alcanzar un rendimiento superior con una cantidad relativamente limitada de datos y recursos computacionales en comparación con entrenar una red profunda desde cero, lo que resalta su eficiencia y escalabilidad para futuros proyectos de Machine Learning.

Potencial de Aplicación Empresarial: La alta precisión del modelo lo hace idóneo para su integración en sistemas inteligentes que requieran análisis predictivos de imágenes. Específicamente, en el contexto de la vigilancia fitosanitaria, la detección temprana y automática de enfermedades es de inmenso valor, ya que permite a las empresas agrícolas:

- Optimizar el uso de recursos (ej. agroquímicos).

- Reducir pérdidas económicas al prevenir la propagación de problemas.

- Mejorar la sostenibilidad de las operaciones agrícolas.

- Asegurar la calidad del producto, lo cual es vital para el cumplimiento de estándares de mercado y contratos de exportación.

En resumen, este proyecto ha entregado una solución de Machine Learning de alto rendimiento y gran valor estratégico, con un potencial significativo para impactar positivamente las operaciones empresariales a través de la predicción precisa y la optimización de procesos.

## Lecciones aprendidas

El principal desafio frente al preprocesamiento de los datos se da para el este tipo de modelos que esperaran un formato de imagen específico (tamaño, canales, normalización). La tubería de despliegue debe asegurar que los datos de entrada se preprocesen correctamente para que coincidan con los requisitos del modelo entrenado.

Para el modelo elegido de fine-tuning aunque es poderoso, existe el riesgo de sobreajuste si no se maneja bien, especialmente con conjuntos de datos pequeños como es este caso, para esto se trata la data con aumento de datos, pero puede persistir el riesgo.

La implemetación de de versionamiento para el datasets y los modelos entrenados en DVC y MLflow furon un desafio inicial en el desarrollo del proyecto pero que fueron superadas al final en la etapa de desarrollo del proyecto.

Aunque el desarrollo inicial fue suficiente con los recursos de Colab, para cargas de trabajo intensivas o requisitos de baja latencia en producción, se recomienda encarecidamente una GPU NVIDIA de la serie Ampere o Hopper (ej., NVIDIA A10, A100 o H100), con un mínimo de 16 GB de VRAM. La aceleración por GPU reduce drásticamente el tiempo de inferencia de la CNN, siendo crítica para entornos de alto tráfico. Para volúmenes de inferencia bajos, una configuración solo con CPU es factible.
Limitaciones: La sesión de Colab es efímera y requiere intervención manual para mantenerse activa, carece de alta disponibilidad, escalabilidad inherente, monitoreo de producción y mecanismos de seguridad robustos.

Para futuros proyecto se tendra en cuenta el correcto versionamiento del código, la data para y los modelos evaluados para establecer una correcta línea de trabajo y establecer la solución más eficiente, asegurando la reproductividad de los experimentos y resultados. 

El monitoreo del rendimiento del modelo (precisión, latencia) y la deriva de datos (data drift) o deriva de concepto (concept drift) seria el siguiente paso a seguir despues de la implementación al usuario del modelo final, dado que los modelos se degradan con el tiempo a medida que el mundo real cambia.

## Impacto del proyecto

Los principales beneficiarios de este proyecto son los productores agrícolas con grandes extensiones de cultivo, quienes enfrentan importantes desafíos al intentar monitorear de forma constante y precisa el estado de salud de sus plantas. La detección oportuna de enfermedades suele ser difícil debido a la magnitud de las áreas sembradas, lo que incrementa el riesgo de pérdidas económicas por propagación no controlada de plagas o infecciones.

El proyecto se encuentra en el dominio de la agricultura de precisión, donde convergen disciplinas como agronomía, visión por computador, Modelos de reconocimiento de imagénes y tecnología de sensores remotos, particularmente mediante el uso de drones.

Los agricultores se enfrentan al reto constante de mantener la sanidad y productividad de sus plantaciones, para esto se deben considerar múltiples factores agronómicos y ambientales, como la fertilidad del suelo, nivel de humedad, pronóstico del tiempo y el control de plagas y enfermedades. Sin embargo, realizar este monitoreo de manera continua y precisa en superficies amplias resulta sumamente costoso, consume mucho tiempo y requiere personal capacitado. En la práctica, esto significa que muchas enfermedades son detectadas tarde, cuando ya han causado daños significativos.

La falta de sistemas eficientes de detección y respuesta oportuna compromete no solo la salud de las plantas, sino también la sostenibilidad económica de las fincas. En cultivos de alto valor como el café, las enfermedades pueden reducir la productividad hasta en un 40% si no se controlan a tiempo, lo que genera pérdidas millonarias, incrementando el uso de agroquímicos y puede comprometer contratos de exportación que requieren estándares de calidad altos.(Atakishiyev et al., 2023).


## Conclusiones

Durante el desarrollo del proyecto se entrenaron modelos con arquitecturas sencillas construidas desde cero, los cuales ofrecieron resultados aceptables. No obstante, los modelos preentrenados demostraron un rendimiento superior. En particular, la aplicación de fine-tuning sobre la arquitectura EfficientNetB0 se consolidó como la solución más precisa y robusta, alcanzando una precisión general del 98 % en el conjunto de validación. Se destacó especialmente con una precisión perfecta (100 %) en la detección de Mildiu Polvoriento y Óxido, y un 94 % para plantas saludables. Estos resultados evidencian la efectividad de los modelos preentrenados en tareas de clasificación de imágenes, especialmente en escenarios con conjuntos de datos limitados.

El empleo de técnicas como el aumento de datos (data augmentation) y el redimensionamiento de imágenes fue fundamental para mejorar la capacidad de generalización del modelo y reducir el riesgo de sobreajuste, lo que permitió construir un sistema confiable a partir de datos relativamente escasos.

En cuanto al despliegue, Google Colaboratory se utilizó como entorno de validación y pruebas iniciales para la API desarrollada con FastAPI, ofreciendo agilidad durante las fases de experimentación. Sin embargo, para su implementación en ambientes productivos se contempla la creación de un Dockerfile que permita contenerizar tanto la aplicación como el modelo y sus dependencias. Esta estrategia garantiza mayor portabilidad, facilidad de mantenimiento y escalabilidad, además de preparar el sistema para su ejecución en infraestructura local o en la nube.

El proceso de despliegue también incluyó consideraciones fundamentales en materia de seguridad y operación, tales como la autenticación mediante claves API, la configuración a través de variables de entorno, el registro de accesos y eventos (logging), y el monitoreo de vulnerabilidades en las dependencias del proyecto. Asimismo, se establecieron los requerimientos mínimos de hardware y software necesarios para garantizar un desempeño óptimo, incluso en entornos con altos volúmenes de solicitudes.

Finalmente, el uso de Git como sistema de control de versiones resultó clave para mantener una estructura de desarrollo ordenada y eficiente. Esta herramienta permitió una colaboración fluida entre los miembros del equipo y aseguró la trazabilidad de los cambios realizados, fortaleciendo la reproducibilidad y calidad del trabajo desarrollado.

## Agradecimientos

- Agradecimientos al equipo de trabajo y a los colaboradores que hicieron posible este proyecto.

Daniel Tejada Hernández
Fredy Esneyder Guzmán Sanchéz
Temis Parra Hernández
Estefanía Puerta Uribe

- Agradecimientos especiales a los patrocinadores y financiadores del proyecto.

Jorge Eliécer Camargo Mendoza, PhD - Aprobador y guía del proceso
Juan Sebastián Lara Ramírez - Aprobador y guía del proceso
