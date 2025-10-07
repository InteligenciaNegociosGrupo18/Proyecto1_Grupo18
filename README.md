## Proyecto 1 – Analítica de Textos (ODS – UNFPA)

Este proyecto implementa un modelo de analítica de textos para clasificar automáticamente opiniones ciudadanas en tres Objetivos de Desarrollo Sostenible (ODS):

* ODS 1: Fin de la pobreza

* ODS 3: Salud y bienestar

* ODS 4: Educación de calidad

El trabajo se desarrolla en el marco de la Agenda 2030 de Naciones Unidas y busca apoyar al UNFPA en la identificación de problemáticas sociales a partir de la voz ciudadana.

Objetivo:

* Aplicar técnicas de procesamiento de lenguaje natural (NLP) y machine learning supervisado para:

* Preprocesar y vectorizar textos en español.

* Construir modelos de clasificación (ODS 1, 3 y 4).

* Evaluar su desempeño mediante métricas estándar (precision, recall, F1-score, accuracy).

* Extraer vocabulario relevante por clase para explicar los resultados y apoyar estrategias de intervención.

Autores: Isabella Caputi - 202122075 Mario Velasquez - 202020502 Sofia Vasquez - 202123910

## Sección 1. (20%) Documentación del proceso de aprendizaje automático. 
Se encuentra en el repositorio ocmo CanvasML_Proyecto1.pdf: https://github.com/sofiavasqueztoro/Proyecto_Grupo18.git

O este enlace de documento:
https://uniandes-my.sharepoint.com/:w:/r/personal/i_caputi_uniandes_edu_co/Documents/CanvasML%20Proyecto%201.docx?d=w4abd8e8e3b3b4bfbab94f22059d5ae59&csf=1&web=1&e=jjyBjY

## Sección 2. (20%) Entendimiento y preparación de los datos.
En el notebook Final_Proyecto_1.ipynb en el repositorio: https://github.com/sofiavasqueztoro/Proyecto_Grupo18.git

## Sección 3. (20%) Modelado y evaluación.
En el notebook Final_Proyecto_1.ipynb en el repositorio: https://github.com/sofiavasqueztoro/Proyecto_Grupo18.git

BONO de modelo de embeddings: notebook P1_BERT.ipynb en el repositorio: https://github.com/sofiavasqueztoro/Proyecto_Grupo18.git

## Sección 4. (20%) Resultados. 

a. Descripción de los resultados obtenidos, que permita a la organización comprenderlos,
haciendo énfasis en el análisis de las métricas de calidad arrojadas por los modelos
utilizados y cómo aportan en la consecución de los objetivos del negocio.

En el notebook Final_Proyecto_1.ipynb en el repositorio: https://github.com/sofiavasqueztoro/Proyecto_Grupo18.git

b. Incluir el análisis de las palabras identificadas para relacionar las opiniones con los ODS
y posibles estrategias que la organización debe plantear utilizando los resultados
obtenidos en los modelos analíticos y una justificación de por qué esa información es
útil para ellos.

En el notebook Final_Proyecto_1.ipynb en el repositorio: https://github.com/sofiavasqueztoro/Proyecto_Grupo18.git

c. Entregar los datos de prueba compartidos, en formato excel, con una columna adicional
que contiene la etiqueta asignada por el modelo que seleccionaron. Este archivo se
utiliza para determinar el f1-score de su modelo analítico y compararlo con los de los
diferentes grupos para asignar parte de la nota del grupo. 

En el archivo prediccionesRandomForest_prueba.xlsx en el repositorio: https://github.com/sofiavasqueztoro/Proyecto_Grupo18.git

d. Generar un video de máximo 5 minutos explicando su proyecto y resultados. El video
debe estar en la wiki respectiva.

Link del video: https://youtu.be/UvVduBh19RQ

En el repositorio: https://github.com/sofiavasqueztoro/Proyecto_Grupo18.git

O en este link: https://www.canva.com/design/DAGyrtD3Y_Y/zrkBFva_3LasQTYU5w3y7w/edit?utm_content=DAGyrtD3Y_Y&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

## Sección 5. (8%) Trabajo en equipo

# Roles y Tareas  

- **Mario Velásquez – Líder de Proyecto e Ingeniero de software responsable de desarrollar la aplicación final (10 horas)**  
  - **Rol**: Como líder de proyecto, se encargó de la gestión general del trabajo, definiendo fechas de reuniones, coordinando las tareas de los integrantes y asegurando el cumplimiento de los entregables. También tuvo la última palabra en decisiones grupales y fue responsable de subir la entrega final. Ademas desarollo la aplicacion final (front end)
  - **Tareas realizadas**: Desarrollo de un modelo de Bayes-Ingenuo de analítica de textos, redacción de las conclusiones a partir de los resultados obtenidos del modelo escogido y desarrollo del front end.
  - **Retos enfrentados**: Garantizar la coherencia entre los resultados técnicos y su adecuada comunicación dentro del informe final.
  - **Uso de ChatGPT**: Apoyo en la interpretación de métricas y apoyo con la selección específica del modelo de Bayes-Ingenuo. 

- **Sofía Vásquez – Ingeniera de Software Responsable del Diseño de la Aplicación y Resultados (10 horas)**  
  - **Rol**: Responsable de la preparación, limpieza y organización de los datos, así como la gestion del proyecto en GitHub.Encargada del diseño de la aplicación y de la generación de los resultados finales 
  - **Tareas realizadas**: Limpieza de los datos, implementación de un modelo de regresion logistica en la analítica de los textos y gestión de la entrega en el repositorio.  
  - **Retos enfrentados**: Manejo de limpieza de los datos y administración del repositorio.  
  - **Uso de ChatGPT**: Apoyo en las técnicas de preprocesamiento, solución de errores en código y ayuda con el modelo de regresion logistica.  

- **Isabella Caputi – Ingeniera de Datos (10 horas)**  
  - **Rol**: Responsable de vincular el trabajo con la estrategia de negocio explicada en el CANVA y la comunicación de resultados. También apoyó en los modelos de analítica y en la gestión del repositorio. Responsable de velar por la calidad del proceso de preparación y automatización de datos para la construcción del modelo analítico.
  - **Tareas realizadas**: Elaboración del **Machine Learning Canvas**, desarrollo de un modelo adicional de Random Forest, realizo el bono con embeddings apoyado con un modelo de BERT y apoyo en la administración de GitHub.  
  - **Retos enfrentados**: Conectar los resultados de los modelos con el CANVA realizado, el bono como tal y coordinar la integración del trabajo grupal.  
  - **Uso de ChatGPT**: apoyo en la estructuración del CANVA, ayuda con los modelos implementados (para Random Forest para la definicion del pipeline y para ajustar los hiperparametros, y especialmente para BERT mucha ayuda en todo el tema de embeddings y utilizacion del modelo como tal).  

## Distribución de Puntos (100)  

- Mario Velásquez: **33,33 puntos**  
- Sofía Vásquez: **33,33 puntos**  
- Isabella Caputi: **33,33 puntos**  

## Reuniones Etapa 1

1. **Reunión de lanzamiento y planeación** (02 de septiembre de 2025):  
   Definición de roles y asignación inicial de tareas.  

2. **Reunión de ideación** (05 de septiembre de 2025):  
   Discusión sobre los enfoques de modelado y alineación con los ODS.  

3. **Reuniones de seguimiento** (no se realizo, se comunico todo por whatsapp):  
   Sesiones breves para coordinar avances y resolver dificultades.  

4. **Reunión de finalización** (12 de septiembre de 2025):  
   Consolidación de resultados, revisión de modelos y redacción de la entrega final en el Github.

## Reuniones Etapa 2

1. **Reunión de lanzamiento y planeación** (29 de septiembre de 2025):  
   Definición de roles y asignación inicial de tareas.  

2. **Reunión de ideación** (30 de septiembre de 2025):  
   Discusión sobre los enfoques de modelado y alineación con los ODS.  

3. **Reuniones de seguimiento** (no se realizo, se comunico todo por whatsapp):  
   Sesiones breves para coordinar avances y resolver dificultades.  

4. **Reunión de finalización** (10 de octubre de 2025):  
   Consolidación de resultados, revisión de modelos y redacción de la entrega final en el Github. 

## Puntos a Mejorar  

- Optimizar la planificación del tiempo para mejorar el manejo del tiempo
- Mejor comunicación con el grupo y realizar mas reuniones de seguimiento 

