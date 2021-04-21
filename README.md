# Proyecto URL-Classifier


## Módulo Generador de modelo de predicción.


> Entrada de datos

- Conjunto de urls (dataset)

    - Proveniente de:
        - https://urlhaus.abuse.ch/api/
    - Ficheros csv con urls de muestras



> Procesamiento de los datos y generacion del modelo de clasificación

- Extración de las caracteristicas para generar el modelo
- Uso de librerias de ML (scikit-learn)

> Salida de los datos

- Fichero pickle que contiene el obejto de modelo para ser integrado en otras plataformas.
