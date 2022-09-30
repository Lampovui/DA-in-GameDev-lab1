# Лабораторная работа. Конструкции языка Python на примере реализации линейной регрессии
Отчет по лабораторной работе #1 выполнил(а):
- Фоменко Андрей Васильевич
- РИ000024
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Ознакомиться с основными операторами зыка Python на примере реализации линейной регрессии.


## Задание 1

Python - Google.colab - Hello World!

![Снимок экрана 2022-09-28 в 18 23 43](https://user-images.githubusercontent.com/83164641/192819726-92afd1da-ffd9-43bc-a283-5ec477c740cb.png)

Unity - Hello World!

<img width="1430" alt="Снимок экрана 2022-09-26 в 11 51 03" src="https://user-images.githubusercontent.com/83164641/192819770-7a8f2065-6a93-476e-8ff9-ac3a4cf78f3a.png">


## Задание 2
### Пошагово выполнить каждый пункт раздела "ход работы" с описанием и примерами реализации задач
Ход работы:
- Произвести подготовку данных для работы с алгоритмом линейной регрессии. 10 видов данных были установлены случайным образом, и данные находились в линейной зависимости. Данные преобразуются в формат массива, чтобы их можно было вычислить напрямую при использовании умножения и сложения.

```py

In [ ]:
#Import the required modules, numpy for calculation, and Matplotlib for drawing
import numpy as np
import matplotlib.pyplot as plt
#This code is for jupyter Notebook only
%matplotlib inline

# define data, and change list to array
x = [3,21,22,34,54,34,55,67,89,99]
x = np.array(x)
y = [2,22,24,65,79,82,55,130,150,199]
y = np.array(y)

#Show the effect of a scatter plot
plt.scatter(x,y)

```

- Определите связанные функции. Функция модели: определяет модель линейной регрессии wx+b. Функция потерь: функция потерь среднеквадратичной ошибки. Функция оптимизации: метод градиентного спуска для нахождения частных производных w и b.

```py
#The basic linear regression model is wx+ b, and since this is a two-dimensional space, the model is ax+ b

def model(a, b, x):
    return a*x + b

#Tahe most commonly used loss function of linear regression model is the loss function of mean variance difference
def loss_function(a, b, x, y):
    num = len(x)
    prediction=model(a,b,x)
    return (0.5/num) * (np.square(prediction-y)).sum()

#The optimization function mainly USES partial derivatives to update two parameters a and b
def optimize(a,b,x,y):
    num = len(x)
    prediction = model(a,b,x)
    #Update the values of A and B by finding the partial derivatives of the loss function on a and b
    da = (1.0/num) * ((prediction -y)*x).sum()
    db = (1.0/num) * ((prediction -y).sum())
    a = a - Lr*da
    b = b - Lr*db
    return a, b
```

 - Начать итерацию
	Инициализация и модель итеративной оптимизации
```py	

#Initialize parameters and display
a = np.random.rand(1)
print(a)
b = np.random.rand(1)
print(b)
Lr = 0.000001
#For the first iteration, the parameter values, losses, and visualization after the
iteration are displayed
a,b = iterate(a,b,x,y,1)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```
![Снимок экрана 2022-09-30 в 11 26 58](https://user-images.githubusercontent.com/83164641/193227020-071c7b69-da9d-4c81-ac70-ac7a05e51ceb.png)

 - На второй итерации отображаются значения параметров, значения
потерь и эффекты визуализации после итерации

```py

a,b = iterate(a,b,x,y,2)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```
![Снимок экрана 2022-09-30 в 11 29 00](https://user-images.githubusercontent.com/83164641/193227443-5ee6bbed-5e92-4a7c-9ab4-905648516442.png)

 - Третья итерация показывает значения параметров, значения потерь и
визуализацию после итерации

```py

a,b = iterate(a,b,x,y,3)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```
![Снимок экрана 2022-09-30 в 11 31 20](https://user-images.githubusercontent.com/83164641/193227898-4cb2fdb6-8fd9-4231-942b-8e5483fa9684.png)

 - На четвертой итерации отображаются значения параметров, значения
потерь и эффекты визуализации

```py

a,b = iterate(a,b,x,y,4)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```
![Снимок экрана 2022-09-30 в 11 33 37](https://user-images.githubusercontent.com/83164641/193228328-99937ffc-9225-4bb0-a085-8eedd17339be.png)

 - Пятая итерация показывает значение параметра, значение потерь и
эффект визуализации после итерации

```py

a,b = iterate(a,b,x,y,5)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```
![Снимок экрана 2022-09-30 в 11 35 01](https://user-images.githubusercontent.com/83164641/193228593-3b2102d4-4d97-4f9b-a162-6836255438ca.png)

 - 10000-я итерация, показывающая значения параметров, потери и
визуализацию после итерации

```py

a,b = iterate(a,b,x,y,10000)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```
![Снимок экрана 2022-09-30 в 11 36 23](https://user-images.githubusercontent.com/83164641/193228854-afbd4943-731b-44ae-b96b-45431fcf71de.png)


## Задание 3.1
### Должна ли величина loss стремиться к нулю при изменении исходных данных? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ.

Loss значительно приближается к нулю при увеличении количества итераций например, если сравнить loss при одной итерации

![Снимок экрана 2022-09-30 в 11 26 58](https://user-images.githubusercontent.com/83164641/193227020-071c7b69-da9d-4c81-ac70-ac7a05e51ceb.png)

И при 10000 итерации

![Снимок экрана 2022-09-30 в 11 36 23](https://user-images.githubusercontent.com/83164641/193228854-afbd4943-731b-44ae-b96b-45431fcf71de.png)

## Задание 3.2
### Какова роль параметра Lr? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ. В качестве эксперимента можете изменить значение параметра.

- При изменении параметра Lr до 0.0006 мы получаем значительное понижение Loss

```py

#Initialize parameters and display
a = np.random.rand(1)
print(a)
b = np.random.rand(1)
print(b)
Lr = 0.0006

```
![Снимок экрана 2022-09-30 в 11 49 07](https://user-images.githubusercontent.com/83164641/193231696-1df788d5-a37e-48a4-8f3f-8c0b571539ca.png)


## Выводы
 Абзац умных слов о том, что было сделано и что было узнано.
 
 


| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
