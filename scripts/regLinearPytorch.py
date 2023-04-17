import pandas as pd
import torch
import torch.nn as nn

data = pd.read_csv("FuelConsumption.csv")
x = data[["CYLINDERS"]].values
y = data["CO2EMISSIONS"].values

# Convertendo os nossos dados para tensores
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

# Definindo modelo
model = nn.Linear(1, 1)

# Definindo otimizadores e metodos de avaliacao
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10000):
    y_pred = model(x_tensor)
    loss = criterion(y_pred, y_tensor.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss {loss.item()}")

# Realizar previsoes
x_new_tensor = torch.tensor([[6.0], [7.0], [8.0], [9.9]])
y_new_tensor = model(x_new_tensor)

# Converter as nossas previsoes em arrays
y_new = y_new_tensor.detach().numpy()

# Mostrar resultados
print(f"Coeficiente: {model.weight.detach().numpy()}")
print(f"Intercepto: {model.bias.detach().numpy()}")
print(f"Previsão: {y_new.squeeze()}")