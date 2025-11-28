# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 17:38:29 2025

@author: Gr
"""
#libs
import torch 
import torch.nn as nn 
import numpy as np
import pandas as pd

#define class
class NNet(nn.Module):
    # для инициализации сети на вход нужно подать размеры (количество нейронов) входного, скрытого и выходного слоев
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        # nn.Sequential - контейнер модулей
        # он последовательно объединяет слои и позволяет запускать их одновременно
        self.layers = nn.Sequential(nn.Linear(in_size, hidden_size), # слой линейных сумматоров
                                    nn.Sigmoid(),                       # функция активации
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.Sigmoid(),
                                    nn.Linear(hidden_size, hidden_size), 
                                    nn.Sigmoid(), 
                                    nn.Linear(hidden_size, out_size), 
                                    nn.Sigmoid(),
                                    )
    # прямой проход    
    def forward(self,X):
        pred = self.layers(X)
        return pred

# X - признаки
# y - правильные ответы, их кодируем числами
# X и y преобразуем в pytorch тензоры
df = pd.read_csv('dataset_simple.csv')
X = torch.Tensor(df.iloc[:, 0:2].values)
y = df.iloc[:, 2].values
y = torch.Tensor(np.where(y == 0, 0, 1).reshape(-1,1))

# Размер входного слоя - это количество признаков в задаче, т.е. количество 
# столбцов в X.
inputSize = X.shape[1] # количество признаков задачи 
# Размер (количество нейронов) в скрытом слое задается нами, четких правил как выбрать
# этот параметр нет, это открытая проблема в нейронных сетях.
# Но есть общий принцип - чем сложнее зависимость (разделяющая поверхность), 
# тем больше нейронов должно быть в скрытом слое.
hiddenSizes = 300 #  число нейронов скрытого слоя 

# Количество выходных нейронов равно количеству классов задачи.
# Но для двухклассовой классификации можно задать как один, так и два выходных нейрона.
outputSize = 1

# Создаем экземпляр нашей сети
net = NNet(inputSize,hiddenSizes,outputSize)
# Веса нашей сети содержатся в net.parameters() 
for param in net.parameters():
    print(param)

# Можно вывести их с названиями
for name, param in net.named_parameters():
    print(name, param)

# Посчитаем ошибку нашего не обученного алгоритма
# градиенты нужны только для обучения, тут их можно отключить, 
# это немного ускорит вычисления
with torch.no_grad():
    pred = net.forward(X)

# Так как наша сеть предсказывает числа от -1 до 1, то ее ответы нужно привести 
# к значениям меток
pred = torch.Tensor(np.where(pred >=0, 0, 1).reshape(-1,1))

# Считаем количество ошибочно классифицированных примеров
err = sum(abs(y-pred))/2
print(err) # до обучения сеть работает случайно, как бросание монетки

# Для обучения нам понадобится выбрать функцию вычисления ошибки
lossFn = nn.MSELoss()

# и алгоритм оптимизации весов
# при создании оптимизатора в него передаем настраиваемые параметры сети (веса)
# и скорость обучения
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# В цикле обучения "прогоняем" обучающую выборку
# X - признаки
# y - правильные ответы
# epohs - количество итераций обучения

epohs = 100
for i in range(0,epohs):
    pred = net.forward(X)   #  прямой проход - делаем предсказания
    loss = lossFn(pred, y)  #  считаем ошибу 
    optimizer.zero_grad()   #  обнуляем градиенты 
    loss.backward()
    optimizer.step()
    if i%10==0:
       print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())

    
# Посчитаем ошибку после обучения
with torch.no_grad():
    pred = net.forward(X)

pred = torch.Tensor(np.where(pred >=0, 0, 1).reshape(-1,1))
err = sum(abs(y-pred))/2
print('\nОшибка (количество несовпавших ответов): ')
print(err) # обучение работает, не делает ошибок или делает их достаточно мало

