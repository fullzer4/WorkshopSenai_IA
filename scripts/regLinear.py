import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv("FuelConsumption.csv")

x = data[["CYLINDERS"]]
y = data["CO2EMISSIONS"]

# Criação do modelo de regressão linear
model = LinearRegression()

# Treinamento do modelo
model.fit(x, y)

# Previsão de novos valores
x_new = np.array([9.9]).reshape((-1, 1))  # transformando x_new em uma matriz bidimensional
y_new = model.predict(x_new)

# Impressão dos resultados
print("Coeficiente: ", model.coef_)
print("Intercepto: ", model.intercept_)
print("Previsão: ", y_new)