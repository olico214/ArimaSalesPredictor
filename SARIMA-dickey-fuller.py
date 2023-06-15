
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import pmdarima as pm
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Cargar datos desde un archivo CSV
#Debe Tener las columnas Year-Semana-TPesos
df = pd.read_csv(r'', dtype={'Year': str})

# Convertir 'TPesos' a numérico (por si acaso no lo es)
df['TPesos'] = pd.to_numeric(df['TPesos'])
# Manejar posibles valores NaN en 'TPesos'
df = df.fillna(0)

# Convertir la columna 'TPesos' a una serie temporal
ts = pd.Series(df['TPesos'].values, index=pd.to_datetime(df['Year'] + df['Semana'].astype(str) + '1', format='%Y%W%w'))

# Plot de la serie temporal
ts.plot()
#Activar para mostrar la primara grafica: plt.show()

# Realizar prueba de Dickey-Fuller
result = adfuller(ts)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# Ajustar el modelo ARIMA con un valor más alto para max_order
model = pm.auto_arima(ts, seasonal=True, m=52, trace=True, max_order=20)




# Imprimir los mejores parámetros del modelo
print('Best ARIMA non-seasonal parameters: ', model.order)
print('Best ARIMA seasonal parameters: ', model.seasonal_order)

p, d, q = model.order
P, D, Q, m = model.seasonal_order

# Ajustamos el modelo SARIMA a los datos
model = SARIMAX(ts, order=(p,d,q), seasonal_order=(P,D,Q,m))
model_fit = model.fit()

# Imprimimos un resumen del modelo ajustado
print(model_fit.summary())

# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
#Activar para mostrar grafica de datos residuales: plt.show()

# Perform Ljung-Box test
ljung_box = sm.stats.acorr_ljungbox(model_fit.resid, lags=[10], return_df=True)
print("Ljung-Box test:")
print(ljung_box)

# Realizar predicciones dentro de la muestra
in_sample_preds = model_fit.predict(start=0, end=len(ts))

# Desplaza las predicciones un paso adelante
in_sample_preds = in_sample_preds.shift(-1)



# Realizar predicciones fuera de la muestra
out_of_sample_preds = model_fit.get_forecast(steps=4).predicted_mean


# Calcular la media de las dos series
average_series = (ts + in_sample_preds) / 2


ventas_predichas_d_Meta_de_ventas = out_of_sample_preds * 1.025

ventas_predichas_df_Meta_esperada= out_of_sample_preds * 1.05
ventas_predichas_df_Meta_estirada= out_of_sample_preds * 1.10

# Trazar las predicciones dentro de la muestra junto con la media
plt.figure(figsize=(12,5), dpi=100)
plt.plot(ts, marker='.', label='Original')
plt.plot(in_sample_preds, marker='.', color='red', label='Predicción')
plt.plot(out_of_sample_preds, marker='.', color='orange', label='Ventas futuras Base')
plt.plot(ventas_predichas_d_Meta_de_ventas, marker='.', color='green', label='Ventas futuras Base meta')
plt.plot(ventas_predichas_df_Meta_esperada, marker='.', color='blue', label='Ventas futuras esperada meta')
plt.plot(ventas_predichas_df_Meta_estirada, marker='.', color='gray', label='Ventas futuras pico meta')
plt.plot(average_series,  color='green', label='Media')
plt.legend(loc='best')
plt.title('Predicción y Media')

# Mostrar las últimas 4 filas del original
print('Últimas 4 semanas de ventas:')
print(ts.tail(4))


i = 0
while i<2:
    print("##########################################", end='\t' )
    i = i+1
print()
# Mostrar las futuras 4 predicciones
print('Futuras 4 predicciones:          ', end='')
for prediccion in out_of_sample_preds.round(2):
    print(prediccion, end='\t')
print()

# Mostrar las futuras 4 predicciones objetivo
print('Futuras 4 predicciones objetivo: ', end='')
for prediccion_objetivo in ventas_predichas_d_Meta_de_ventas.round(2):
    print(prediccion_objetivo, end='\t')
print()

# Mostrar las futuras 4 predicciones objetivo esperado
print('Futuras 4 predicciones objetivo esperado: ', end='')
for prediccion_esperada in ventas_predichas_df_Meta_esperada.round(2):
    print(prediccion_esperada, end='\t')
print()

# Mostrar las futuras 4 predicciones objetivo pico
print('Futuras 4 predicciones objetivo pico:     ', end='')
for prediccion_pico in ventas_predichas_df_Meta_estirada.round(2):
    print(prediccion_pico, end='\t')
print()

i = 0
while i<2:
    print("##########################################", end='\t' )
    i = i+1
print()
plt.show()


