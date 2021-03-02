import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Acidentes do trabalho por idade  simplificado
dataset = pd.read_csv ('dados_aci.csv',encoding = "ISO-8859-1")
#dataset = pd.read_csv ('dados.csv')
# pega dados apartir de 2018, das colunas idade e qtd de acidentes
y = dataset.iloc[:, 0:1].values
x = dataset.iloc[:, 1:2].values

print(x)

print(y)

x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size=0.15,random_state=0)
linearRegression = LinearRegression()

linearRegression.fit(x_treinamento, y_treinamento)

y_pred = linearRegression.fit(x_treinamento, y_treinamento)

equa = f'{linearRegression.coef_[0]}x + {linearRegression.intercept_:}'

plt.scatter(x_treinamento, y_treinamento, color="red")
plt.plot(x_treinamento,linearRegression.predict(x_treinamento),color="blue")
plt.title(f'Idade x qtd de acidentes| equação {equa} ')
plt.xlabel("qtd de acidenes")
plt.ylabel("idade")
plt.show()

