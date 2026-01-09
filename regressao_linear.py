import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

path = "C:\\Users\\Pichau\\OneDrive\\Estudos\\Algoritmos\\dataset\\Linear_Regression.csv"

data = pd.read_csv(path)

print(data.head())

X = data[['X']]
y = data[['Y']]

model = LinearRegression()

model.fit(X, y)

print("Coeficientes:", model.coef_)
print("Intercepto:", model.intercept_)

# Fazer previsões (opcional)
y_pred = model.predict(X)
print("Previsões:", y_pred[:5])

plt.plot(X, y_pred, color='red', label='Previsão')
plt.show()
