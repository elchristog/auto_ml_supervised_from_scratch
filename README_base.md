# Documentación: Modelo de propensión para 

A continuación describimos el proceso, versionado, fuentes y hallazgos en el desarrollo del modelo de propensión para .

```
|-- Introducción
|-- Objetivo y OKR
|-- Fuentes
|-- ETL
|-- Version 0.1
|   |-- Variables incorporadas:
|   |-- Funciones generadas:
|   |-- Modelo ajustado:
|   |-- Métricas:
|   |-- Hallazgos:
|-- Version 0.2
|   |-- Variables incorporadas:
|   |-- Funciones generadas:
|   |-- Modelo ajustado:
|   |-- Métricas:
|   |-- Hallazgos:
|-- Version 0.3
|   |-- Variables incorporadas:
|   |-- Funciones generadas:
|   |-- Modelo ajustado:
|   |-- Métricas:
|   |-- Hallazgos:
|-- Pipeline
|-- Despliegue
|-- Transición al DataLake
```

## Introducción

Este modelo busca modelar la propensión de compra para clientes de la `Banca `, permitiendo al área comercial priorizar la gestión de los clientes enfocandose en aquellos que sean más propensos a realizar altas en productos del banco y logrando aumentar en 8% los desembolsos mensuales para la Banca [1].

Los repositorios para el entrenamiento [Model Build](https://git-codecommit.us-east-1.amazonaws.com/v1/repos/sagemaker-sagemaker--endpoint-p-tqi7sbliff3r-modelbuild), Despliegue  [Model Deploy](https://git-codecommit.us-east-1.amazonaws.com/v1/repos/sagemaker-sagemaker--endpoint-p-tqi7sbliff3r-modeldeploy) y monitoreo [Model Monitor](https://git-codecommit.us-east-1.amazonaws.com/v1/repos/sagemaker-sagemaker--endpoint-p-tqi7sbliff3r-modelmonitor) del modelo se encuentran alojados en CodeCommit y son gestionados mediante los pipelines de SageMaker. 

## Objetivo y OKR

El objetivo es aumentar en 8% los desembolsos mensuales para la Banca .

## Fuentes

Se han considerado variables demográficas, variables de riesgos, saldos, principalidad y otras variables de características. Dichas variables originalmente son obtenidas mediante el área de  y cargadas a S3, pero, cuando la data se encuentre cargada al DataLake, se procede a realizar una lectura más directa de las fuentes [2].

Fuentes originales recibidas en formato CSV y cargadas a S3:

    * [1] Tablón: Tabla con información mensual de clientes y contratos activos, tanto del pasivo como del activo, esta tabla tiene además el tipo de producto, gerente, segmento, saldo punta, saldo medio, producto bancario al cierre de cada mes.
    * [2] Clientes: Esta tabla está a nivel de cliente y tiene información del segmento del cliente y si ha presentado cambios, sector económico, fecha de ingreso, ciudad y pais, volumen del activo y del pasivo, grupo económico interno GEVC. 
    * [3] Riesgos por cliente: Información mensual por cliente donde encontramos el estado(sujeto/No sujeto de crédito), la clasificación del sector dada por riesgos, los rating, calificaciones de riesgos (, Chile y Colombia), encontramos además los cupos otorgados para diversas, operaciones de leasing, operaciones específicas y mesa, también encontramos la clasificación WL. 
    * [4] Sector CIFIN: Información de endeudamiento del cliente en el sector externo, esta base se registra de manera trimestral, el cliente está repetido tantas veces como deudas tenga con otras entidades. 
    * [5] CIIU: Códigos del sector económico registrados por el dane.
    * [6] Mapa productos: Base de productos, donde se relaciona si hace parte del activo/pasivo, la familia del producto y la subfamilia.
    * [7] Ejecutivos únicos: Base de ejecutivos con estado de activo/cerrado, donde se relaciona nombre y código del gerente y código de la cartera asignada, director y segmento. 
    * [8] Director: Base de directores con estado de activo/cerrado, donde se relaciona nombre y código del director y la ciudad que gestiona.
    * [9] Gerentes: Base de gerentes con estado de activo/cerrado, donde se relaciona nombre y código del gerente. 
    * [10] Bancas: Base de la clasificación de la banca de la compañía: Corporate, Grandes Empresas, Gobierno, etc. Nombre del segmento y del subsegmento y el estado actual de la banca (Activo/Cerrado).
    * [11] Invoice Number: Base construida a partir de los contratos, con los conteos de eventos de altas por mes y por cliente de créditos del activo (ordinaria, leasing financiero y moneda extranjera).
    * [12] Info Contratos: Base histórica de contratos tanto del activo como del pasivo, con fecha del alta, fecha de vencimiento, número de contrato, número de identificación del cliente e importe inicial del contrato.
    * [13] EMIS: Información financiera del cliente donde se reporta utilidades, ganancia operativa, activos, pasivos, deuda, relación deuda-capital.

## ETL

El proceso de transformación de datos se planteó y ejecuto con el propósito de minimizar el impacto en la transición de datos entre las fuentes y el DataLake, para esto se utilizó S3 para el almacenamiento de los CSV originales, Glue Jobs para la transformación de los CSV a Parquet, la limpieza de registros, generación de variables y transformaciones correspondientes, y la carga de las variables a S3. Athena para la lectura y análisis Ad Hoc de las variables, SageMaker Data Wrangler para la transformación de Machine Learnind de lops datos, SageMaker para el entrenamiento del modelo, y SageMaker Pipelines para la industrialización del modelo.

Ahora, enfocandonos en el proceso de ETL, cada tabla fue transformada mediante un Glue Job independiente donde la lectura se asoción con un nodo apuntando al origen en S3 para que en la migración al DataLake se pueda realizar una lectura directa de las fuentes cambiando el nodo de origen y realizando cambios sustanciales en el proceso de transformación.

La Escritura en Parquet de cada Glue Job  se realizó en una carpeta independiente en S3 y se creó un Crawler independiente para cada tabla, de esta manera, esta salida mapeada por el Crawler será la misma independientemente si la ETL trabaja con tablas cargadas a S3 o DataLake.

## Version 0.1
Esta versión del modelo buscaba modelar la propensión de compra para clientes de la `Banca ` utilizando un primer grupo de variables muy generales.

### Variables incorporadas

    * `fecha_data_filtro`: Fecha de la data de filtro.
    * `label`: Variable de salida.
    * `T2_Bogota`: Variable indicadora para empresas que pertenecen a la Tier2 de Bogotá.
    * `T2_Regiones`: Variable indicadora para empresas que pertenecen a la Tier2 de las regiones.
    * `T3_Bogota`: Variable indicadora para empresas que pertenecen a la Tier3 de Bogotá.
    * `T3_Regiones`: Variable indicadora para empresas que pertenecen a la Tier3 de las regiones.
    * `antiguedad_meses`: Variable indicadora de antiguedad en meses de la empresa con el Banco.
    * `tiene_pasivo`: Variable indicadora de si la empresa tiene productos del pasivo.
    * `saldo_punta_pasivo`: Saldo punta del pasivo.

### Funciones generadas

De cara al Pipeline de industrialización del modelo, se generaron funciones para la transformación de los datos y ajuste del modelo:

* `read_tablon`: La función read_tablon lee usando Athena el tablón procesado por ETL.
* `read_invoice_number`: La función read_invoice_number lee usando Athena el invoice_number procesado por ETL.
* `read__demografico`: La función read__demografico lee usando Athena el ETL procesado _demografico.
* `read__saldo`: La función read__saldo lee usando Athena el ETL procesado _saldo.
* `merge_data`: La función merge_data combina ocho dataframes.
* `label_creation`: La función label_creation crea una nueva columna con la etiqueta para el dataframe combinado.
* `clean_mes_data`: La función clean_mes_data crea una nueva variable a partir de la variable fecha_data que contiene solo el año y el mes.
* `dummy_segment`: La función dummy_segment transforma la variable sub_segmento_pyf en dummies para cada categoría.
* `discretizer_saldo_activo`: La función discretizer_saldo_activo crear rangos para la columna de saldos del activo.
* `discretizer_saldo_pasivo`: La función discretizer_saldo_pasivo crear rangos para la columna de saldos del pasivo.
* `select_columns`: La función select_columns selecciona las columnas que son necesarias para el modelo.
* `fill_na`: La función fill_na llena las columnas con valores nulos.
* `split_set`: Función que realiza la separación de los datos en entrenamiento y prueba.
* `split_train`: Función que realiza la separación de las variables independientes y variable dependiente.
* `split_test`: Función que realiza la separación de las variables independientes y variable dependiente.
* `split_valid`: Función que realiza la separación de las variables independientes y variable dependiente.
* `standardize_variables`: Función que realiza la normalización de las variables.
* `resample_data`: Función que realiza el resampling de los datos.
* `log_regression`: Función que realiza el ajuste del modelo mediante el algoritmo de regresión logística.
* `prediction_log`: Función que realiza las predicciónes y el cálculo de probabilidades para la muestra de prueba.
* `metrics_model`: Función que realiza el cálculo de las métricas del modelo.
* `confusion_mtx`: Función que genera el gráfico de la matríz de confusión.
* `cum_gain`: Función que genera el gráfico de la curva de ganancia.
* `threshold_plot`: Función que genera el gráfico de la distribución de probabilidades predicha.

Estas funciones fueron orquestadas mediante un Pipeline para su aplicación y llevadas al Pipeline de SageMaker.

### Data Preparation

A continuación detallamos los desarrollos encaminados a la preparación de la data en el desarrollo del modelo.

#### Label ¿Cuál es? - Periodicidad

Hemos definido con el equipo que nuestra avriable label debe indicar con el valor 1 aquellas altas realizadas......

#### Desbalance del label

Luego de generar la variable identificamos una frecuencia de 80% para el caso 0 y de 20% para el 1.

#### Universo dado filtros

...

#### Definición de la métrica, variables sin modelo

...

### Train & Tune

...

#### Buenas prácticas

...

#### Ejecución del modelo inicial

El modelo ajustado en esta versión fue una regresión logística sin parametros de penalización, con el objetivo de hacer un ajuste Ad Hoc y posteriormente ir refinando nuevos modelos y calibrando sus parametros.

### Evaluación de métricas

Esta versión del modelo obtuvo:

* Un Accuracy de 65% 
* Un AUC de 69% 
* Un F1 de 6.8%
* Una Ganancia con el 20% de registros más propensos de 51.2%

### Hallazgos

Pudimos notar que cerca del 65% de los clientes tienen unicamente productos del pasivo, lo que implica que la propensión de compra es menor para este segmento y que la oportunidad de aumentar el Activo es grande al movilizae estos clientes.


[1] OKR propuesto por la Banca .
[2] OKR propuesto por la Banca .
