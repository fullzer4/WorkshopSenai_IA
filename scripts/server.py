import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

data = pd.read_csv("FuelConsumption.csv")

x = data[["CYLINDERS"]]
y = data["CO2EMISSIONS"]

# Criação do modelo de regressão linear
model = LinearRegression()

# Treinamento do modelo
model.fit(x, y)

# Criação do servidor em Flask
app = Flask(__name__)
CORS(app)

# Rota para realizar previsões
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    data = request.get_json()
    x_new = np.array(data['cylinders']).reshape((-1, 1))
    y_new = model.predict(x_new)
    return jsonify({'CO2 EMISSIONS': y_new[0]})

# Execução do servidor Flask
if __name__ == '__main__':
    app.run(debug=True)