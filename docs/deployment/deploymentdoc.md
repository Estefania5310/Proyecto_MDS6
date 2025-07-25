# Despliegue de modelos

## Infraestructura

- **Nombre del modelo:** Modelo de clasificación para seguimiento fitosanitario de plantas

- **Plataforma de despliegue:**

El modelo se ha implementado utilizando una aplicación FastAPI ejecutada en Google Colaboratory. Si bien esta configuración permitió un desarrollo y pruebas rápidos, es importante notar que Google Colaboratory no está diseñado para el despliegue de servicios continuos y en producción. La ejecución en este entorno es temporal y requiere la sesión activa de Colab.

La contenerización mediante Docker se considera un paso fundamental para garantizar la portabilidad y el despliegue consistente. Se desarrollará un Dockerfile que encapsule la aplicación FastAPI, el modelo pre-entrenado y todas sus dependencias. Esto permitirá construir una imagen Docker auto-contenida.

- **Requisitos técnicos:** 

A continuación, se detallan los requisitos técnicos fundamentales para el despliegue y la operación óptima del "Modelo de clasificación para seguimiento fitosanitario de plantas". Esta especificación se basa en el entorno de desarrollo actual en Google Colaboratory.

1. Entorno de Software:

- **Sistema Operativo:** Se requiere un entorno operativo basado en Linux, preferentemente una distribución de servidor estable y con soporte a largo plazo. Las pruebas actuales se realizaron en un entorno con Linux 6.1.123+ (x86_64). Para producción, se recomienda una versión como Ubuntu Server 24.04 LTS (Noble Numbat) o Debian 12 (Bookworm), que aseguran compatibilidad con las librerías de Python y contenedores, además de soporte extendido.

- **Lenguaje de Programación:** Python 3.11.13. Esta es la versión actualmente en uso y es adecuada para la producción, ofreciendo un buen equilibrio entre características modernas y estabilidad.

- **Gestor de Paquetes:** pip (versión 24.x.x o superior), junto con un gestor de entornos virtuales como venv o conda, para asegurar el aislamiento y la gestión precisa de las dependencias.

- **Servidor de Aplicaciones Web: Para la exposición de la API FastAPI, se utilizará Uvicorn (versión 0.35.0).

2. Librerías y Dependencias de Python:

Las siguientes librerías son críticas para la funcionalidad del modelo y la API, y sus versiones actuales, detectadas en el entorno de desarrollo, son adecuadas para un despliegue en producción:

- **TensorFlow:** Versión 2.18.0. Esta versión actual de TensorFlow (que integra Keras 3.8.0) es compatible con el modelo CNN entrenado y ofrece las últimas optimizaciones para inferencia.

- **Keras:** Versión 3.8.0 (integrado en TensorFlow 2.18.0).

- **FastAPI:** Versión 0.116.1. Esta versión es reciente y robusta para la construcción de APIs.

- **Uvicorn:** Versión 0.35.0, compatible con la versión actual de FastAPI.

- **NumPy:** Versión 2.0.2, para operaciones numéricas eficientes.

- **Pydantic:** Versión 2.11.7, fundamental para la validación de datos y la serialización/deserialización en FastAPI.

- *Requests:** Versión 2.32.3, si la aplicación necesita realizar solicitudes HTTP salientes.

3. Requisitos de Hardware (Especificaciones para Producción):

Considerando que el entrenamiento y la predicción fueron eficientes en la máquina por defecto de Google Colaboratory, se establecen los siguientes requisitos mínimos y recomendaciones para una operación en producción:

- **CPU:** Mínimo 4 vCPUs, preferentemente procesadores modernos (ej., Intel Xeon Scalable de 3ra generación o AMD EPYC de 3ra generación o superior). Esto permite el manejo concurrente de solicitudes a la API y la ejecución eficiente de la inferencia del modelo.

- **GPU (Recomendado para Alto Volumen/Baja Latencia):** Aunque el desarrollo inicial fue suficiente con los recursos de Colab, para cargas de trabajo intensivas o requisitos de baja latencia en producción, se recomienda encarecidamente una GPU NVIDIA de la serie Ampere o Hopper (ej., NVIDIA A10, A100 o H100), con un mínimo de 16 GB de VRAM. La aceleración por GPU reduce drásticamente el tiempo de inferencia de la CNN, siendo crítica para entornos de alto tráfico. Para volúmenes de inferencia bajos, una configuración solo con CPU es factible.

- **RAM:** Mínimo 16 GB de RAM. Esta capacidad es suficiente para cargar el modelo en memoria, manejar las operaciones de la API y el preprocesamiento de imágenes sin incurrir en swapping o degradación del rendimiento.

- **Almacenamiento:** Mínimo 50 GB de almacenamiento SSD de alto rendimiento. Esto es crucial para el rápido acceso al código de la aplicación, el modelo pre-entrenado y los logs del sistema. Se prefiere un sistema de archivos robusto y redundante si se despliega en infraestructura propia.

4. Consideraciones de Despliegue Adicionales:

- **Persistencia del Modelo:** El modelo entrenado debe ser accesible por la aplicación de despliegue. Se recomienda almacenarlo en un sistema de almacenamiento de objetos (ej., Google Cloud Storage, AWS S3) o un volumen persistente que pueda ser montado por el contenedor o instancia.

- **Variables de Entorno:** Todas las configuraciones sensibles o específicas del entorno (ej., rutas de archivos del modelo, claves API, configuraciones de red) se gestionarán exclusivamente a través de variables de entorno, no codificadas en el código base.

- **Configuración de Red:** El servicio FastAPI deberá exponerse en puertos estándar (ej., 80 para HTTP, 443 para HTTPS) y se implementarán políticas de firewall adecuadas para limitar el acceso únicamente a los orígenes autorizados.

**Requisitos de seguridad:** 

Aunque el proyecto inicial no priorizó explícitamente los requisitos de seguridad debido a su entorno de desarrollo, es fundamental integrar medidas básicas al transicionar a un despliegue en producción. La implementación de controles de seguridad minimiza los riesgos de acceso no autorizado, manipulación de datos o interrupciones del servicio.

Dada la necesidad de una implementación simple y efectiva, se proponen los siguientes puntos clave:

- **Autenticación Básica con Clave API:** Implementar un mecanismo simple donde el cliente que consume la API del modelo deba incluir una clave API (API Key) predefinida en cada solicitud. Esta clave puede validarse dentro de la aplicación FastAPI. Esto actúa como un primer nivel de control, asegurando que solo los servicios o usuarios que poseen la clave puedan interactuar con el modelo. La clave debe ser un valor complejo y difícil de adivinar.

- **Variables de Entorno para Claves:** La clave API nunca debe estar "hardcodeada" en el código. Debe cargarse de forma segura como una variable de entorno en el entorno de despliegue, permitiendo una fácil rotación y evitando su exposición en el repositorio de código.

- **Vigilancia de Vulnerabilidades:** Utilizar herramientas o servicios que escaneen las dependencias de Python (listadas en requirements.txt) en busca de vulnerabilidades de seguridad conocidas. Esto es crucial, ya que una vulnerabilidad en una librería de terceros podría comprometer todo el servicio.

- **Registros de Acceso:** Configurar el logging de la aplicación FastAPI para registrar intentos de acceso (exitosos y fallidos), errores y eventos relevantes. Aunque no es una medida de seguridad activa, estos registros son vitales para auditorías y para identificar patrones de actividad sospechosa.


**Diagrama de arquitectura:**

A continuación, se describe la arquitectura actual de la solución basada en Google Colaboratory

- **Componentes**
  - Cliente/Usuario: Inicia una solicitud (hipotéticamente, a través de alguna interfaz o script local) hacia la aplicación.
  - Entorno de Google Colaboratory: Actúa como el servidor.
  - Notebook de Python: Contiene el código de la aplicación FastAPI y el modelo CNN de Keras.
  - Servidor FastAPI (Uvicorn): Iniciado dentro del notebook, expone un endpoint de inferencia.
  - Modelo CNN (Keras/TensorFlow): Cargado en la memoria de la sesión de Colab, realiza las predicciones.
  - Internet: Permite la comunicación (generalmente HTTP) entre el cliente y el endpoint temporal de Colab (ej., a través de ngrok o similar para exponer localmente).

- **Flujo de Datos**

![Flujo de datos](https://github.com/Estefania5310/Proyecto_MDS6/blob/main/scripts/deployment/diagrama_arquitectura.png)


  1. Un Cliente/Usuario envía una solicitud HTTP (POST) con datos de imagen al endpoint expuesto por la aplicación FastAPI en Google Colaboratory.
  2. La Aplicación FastAPI recibe la solicitud.
  3. La aplicación preprocesa la imagen de entrada y la pasa al Modelo CNN.
  4. El Modelo CNN realiza la inferencia y devuelve la predicción.
  5. La Aplicación FastAPI formula una respuesta y la envía de vuelta al Cliente/Usuario.

  **Limitaciones:** La sesión de Colab es efímera y requiere intervención manual para mantenerse activa, carece de alta disponibilidad, escalabilidad inherente, monitoreo de producción y mecanismos de seguridad robustos.


## Código de despliegue
  * **Archivo principal:**
    * **Nombre:** `App2`
    * **Ruta dentro del proyecto:** `src/nombre_paquete/evaluation/app2.py`
    * **Función:** Contiene la instancia de FastAPI, la definición de los *endpoints* de la API (ej. `/predict`), la lógica para cargar el modelo y el procesamiento de las solicitudes de inferencia.

  * **Rutas de acceso a los archivos:**
    * **Nombre:** `model.joblib`
    * **Ruta dentro del proyecto:** `src/nombre_paquete/models/best_models/model.joblib`
    * **Función:** Este archivo binario contiene la arquitectura de la red neuronal convolucional (CNN) entrenada y sus pesos, esencial para realizar las predicciones.
    
    
  * **Script de Preprocesamiento de Imágenes:**
    * **Nombre:** `preprocessing_prediction.py`
    * **Ruta dentro del proyecto:** `app/src/nombre_paquete/preprocessing/preprocessing_prediction.py`
    * **Función:** Contiene la función `preprocess_image_for_prediction` encargada de cargar, redimensionar y normalizar las imágenes de entrada, preparándolas para ser utilizadas por el modelo CNN. Este módulo es invocado antes de pasar la imagen al modelo.
  

  * **Script de Interpretación de Predicciones:**
    * **Nombre:** `interpret_prediction.py`
    * **Ruta dentro del proyecto:** `app/src/nombre_paquete/evaluation/interpret_prediction.py`
    * **Función:** Contiene la función `interpret_prediction` que toma las salidas crudas del modelo y las transforma en un formato legible, mostrando las probabilidades por clase y la clasificación final con su confianza. Este módulo es invocado por después de que el modelo haya realizado la inferencia.



## Documentación del despliegue

- **Instrucciones de instalación:** 

- **Instrucciones de configuración:** 
- **Instrucciones de uso:**
- **Instrucciones de mantenimiento:** 
