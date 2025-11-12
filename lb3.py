# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 13:58:49 2025

@author: Groshev
"""
#biblioteki
import torch 
import pandas as pd
import random
import torch.nn as nn


# Основная структура данных pytorch - тензоры
# Основные отличия pytorch-тензоров от numpy массивов:
# 1 Место их хранения можно задать ( память CPU или GPU )
# 2 В отношении тензоров можно задать вычисление и отслеживание градиентов

# Тензор x целочисленного типа, хранящий случайные значение
# Пустой тензор
x = torch.randint(1, 10, (5,3))
print(x)

# Преобразовать тензор к типу float32
x = x.to(dtype=torch.float32)
x.requires_grad = True

# Возвести в степень n = 3
#12//2=0
y=x**3
print(y)
# Умножить на случайное число от 1 до 10
z=y*random.uniform(1, 10)
print(z)
# Взять экспоненту от полученного числа
exp=torch.exp(z)
print(exp)
# Получить значение производной для полученного значения по x
exp.backward(torch.ones_like(exp))
print(x.grad) # градиенты d(ex)/dx


###########                                                        ############
###########    Обучение линейного алгоритма на основе нейронов    #############
###########                                                       #############


# На основе кода обучения линейного алгоритма создать код для решения задачи классификации цветков ириса

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
X = torch.tensor(df.iloc[:, 0:4].values, dtype=torch.float32) 

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_labels = le.fit_transform(df.iloc[:, 4])
y = torch.zeros(150, 3, dtype=torch.float32)
y[torch.arange(150), y_labels] = 1 

linear = nn.Linear(4, 3)
# при создании веса и смещения инициализируются автоматически
print ('w: ', linear.weight)
print ('b: ', linear.bias)

lossFn = nn.MSELoss() 

optimizer = torch.optim.SGD(linear.parameters(), lr=0.01) 

# итерационно повторяем шаги
# в цикле (фактически это и есть алгоритм обучения):
for i in range(0,20):
    optimizer.zero_grad() # забыли
    pred = linear(X)
    loss = lossFn(pred, y)
    print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())
    loss.backward()
    optimizer.step()
    