# Proyecto URL-Sentinel

## Modulo 3 Reporte y resultado de la clasificacion de las URL analisadas

> Entrada de datos

- Solicitud request al API /api/v1/predict
- uvicorn main:app --reload

> Procesamiento de los datos

- Carga el modelo generado por el modulo 2 y realiza las acciones de clasificacion

> salida de los datos

- mostrando como resultado si la url ingresada es M o B
- Genera salva en base de datos de las URl maliciosas
- Envia alerta a los sistemas de monitoreo