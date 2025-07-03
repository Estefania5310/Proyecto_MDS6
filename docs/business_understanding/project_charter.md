[# Project Charter - Entendimiento del Negocio

## Nombre del Proyecto
Clasificación automática de enfermedades en hojas de plantas usando aprendizaje profundo

---

## Objetivo del Proyecto
Desarrollar un modelo de aprendizaje profundo (CNN con transfer learning) que clasifique imágenes de hojas en tres categorías: Healthy, Powdery Mildew y Rust.  
Este proyecto busca mejorar la capacidad de detección temprana de enfermedades en cultivos mediante visión por computador, contribuyendo a reducir pérdidas agrícolas.

---

## Alcance del Proyecto

### Incluye:
- Uso del dataset público de Kaggle que contiene 1.530 imágenes clasificadas por tipo de enfermedad.
- Preprocesamiento, entrenamiento, validación y prueba de modelos de clasificación de imágenes.
- Generación de métricas de desempeño y visualizaciones de resultados.
- Documentación y versionamiento del proyecto con GitHub.

### Resultados esperados:
- Modelo con una precisión mínima del 85% en el conjunto de prueba.
- Matriz de confusión, accuracy y AUC ROC.
- Pipeline reproducible desde Google Colab.

### Criterios de éxito:
- Clasificador entrenado con buen desempeño (mínimo 85% accuracy).
- Repositorio completo, versionado y con documentación clara.
- Presentación clara de hallazgos, visualizaciones y recomendaciones.

### Excluye:

- Predicción en tiempo real con cámaras.

---

## Metodología

Se usará la metodología TDSP (Team Data Science Process), abordando las siguientes fases:
1. Entendimiento del negocio y definición del problema
2. Adquisición y análisis exploratorio de datos
3. Preparación y preprocesamiento
4. Entrenamiento de modelos CNN con transfer learning (MobileNet, VGG o similar)
5. Evaluación de desempeño
6. Documentación y entrega

---

## Cronograma

| Etapa                                      | Duración Estimada | Fechas                  |
|-------------------------------------------|-------------------|--------------------------|
| Entendimiento del negocio y carga de datos| 2 semanas         | 1 de mayo - 15 de mayo   |
| Preprocesamiento y análisis exploratorio  | 4 semanas         | 16 de mayo - 15 de junio |
| Modelamiento y entrenamiento              | 4 semanas         | 16 de junio - 15 de julio|
| Despliegue (entorno reproducible)         | 2 semanas         | 16 de julio - 31 de julio|
| Evaluación y entrega final                | 3 semanas         | 1 de agosto - 21 de agosto|

---

## Equipo del Proyecto

- **Estefanía Puerta Uribe** – Científica de datos
- **Daniel Tejada Hernández** – Científico de datos
- **Fredy Esneyder Guzmán Sánchez** – Científico de datos
- **Temis Parra Hernández** – Científica de datos

---

## Presupuesto
Proyecto académico sin costo económico. Recursos computacionales provistos por Google Colab.  

---

## Stakeholders

- **Docente de Proyecto Aplicado (Universidad Nacional)** – Aprobador y guía del proceso
- **Miembros del equipo** – Desarrolladores e implementadores
- **Comunidad académica** – Receptores de los resultados

**Expectativas**:
- Aplicar correctamente la metodología TDSP
- Entregar resultados funcionales, explicables y documentados

---

## Aprobaciones

- **[Nombre del docente o tutor del curso]** – Docente guía  
- Firma: ___________________  
- Fecha de aprobación: ___________________]