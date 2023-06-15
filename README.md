# Modelo de predicción de ventas utilizando ARIMA

Este repositorio contiene un modelo de predicción de ventas basado en el análisis de series temporales utilizando el enfoque ARIMA (Autoregressive Integrated Moving Average).

El modelo se implementa en Python y utiliza las siguientes bibliotecas:

- pandas: para la manipulación de datos y la creación de la serie temporal.
- numpy: para el manejo de operaciones numéricas.
- statsmodels: para realizar pruebas estadísticas y ajustar el modelo SARIMA.
- pmdarima: para seleccionar automáticamente los mejores parámetros del modelo ARIMA.
- matplotlib: para trazar las gráficas de la serie temporal y los resultados del modelo.

## Cómo utilizar el modelo

1. Asegúrate de tener instaladas las bibliotecas requeridas mencionadas en el archivo `requirements.txt`.
2. Descarga el archivo `modelo_prediccion_ventas.py`.
3. Prepara tus datos en un archivo CSV con las siguientes columnas: 'Year', 'Semana', y 'TPesos'. Asegúrate de tener al menos 3 años de datos semanales.
4. Abre el archivo `modelo_prediccion_ventas.py` y establece la ruta del archivo CSV en la línea correspondiente: `df = pd.read_csv(r'ruta_del_archivo.csv', dtype={'Year': str})`.
5. Ejecuta el script. Obtendrás los resultados de la predicción y gráficas de visualización.

## Resultados del modelo

El modelo realiza los siguientes pasos:

1. Carga los datos de ventas desde un archivo CSV y los convierte en una serie temporal.
2. Realiza una prueba de Dickey-Fuller para verificar la estacionariedad de la serie.
3. Utiliza el método `auto_arima` de la biblioteca `pmdarima` para seleccionar automáticamente los mejores parámetros del modelo ARIMA.
4. Ajusta un modelo SARIMA a los datos utilizando los parámetros seleccionados.
5. Realiza pruebas estadísticas de los residuos del modelo para evaluar su validez.
6. Realiza predicciones dentro y fuera de la muestra utilizando el modelo ajustado.
7. Calcula diferentes escenarios de predicción utilizando factores multiplicativos.
8. Grafica la serie temporal original, las predicciones y los diferentes escenarios de predicción.

## Contribuir

Si deseas contribuir a este modelo de predicción de ventas, puedes seguir estos pasos:

1. Realiza un fork de este repositorio.
2. Crea una rama para tu nueva característica (`git checkout -b nueva-caracteristica`).
3. Realiza los cambios necesarios y haz commit de tus modificaciones (`git commit -am 'Agrega una nueva característica'`).
4. Realiza un push a la rama (`git push origin nueva-caracteristica`).
5. Abre una pull request en este repositorio.

## Licencia

Este proyecto está bajo la Licencia MIT. Puedes consultar el archivo [LICENSE](LICENSE) para obtener más detalles.

