# Construindo uma Rede Neural Artificial

Construindo sua primeira rede neural usando Keras, Theano e Tensorflow
Parte da matéria de DeepLearning da escola de Engenharia de Software da UnB

## Tarefas
* Crie uma branch com seu nome
* Troque a função de ativação
* Compile a rede
* Commit do resultado (NA SUA BRANCH)
* Calcule a acurácia, a média e a variação

## Funções de Ativação Utilizadas

Uma função utilizada para saber a saída do nó, ou seja, tenta determinar a saída da rede neural

> sigmoid: A principal razão pela qual usamos a função sigmóide é porque ela nos fornece uma forte precisão em contextos binários pois varia entre 0 e 1. Portanto, é especialmente usado em modelos onde temos que prever a probabilidade de existência de algo.

![Sigmoid Function Example](https://miro.medium.com/max/485/1*Xu7B5y9gp0iL5ooBj7LtWw.png)

> tanh: Bastante similar a função sigmoid descrita cima, entretanto consegue se dar bem com valores fortemente positivos ou fortemente negativos. 

![Tanh example](https://www.medcalc.org/manual/_help/functions/tanh.png)

## Testes Iniciais

### Primeiro Teste
### Funções de Ativação
```python
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim = 11))

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))  
```
### Resultado Obtido
![3sigmoid](https://imgur.com/YofnqL4.png)

## Segundo Teste
### Funções de Ativação
```python
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))  
```
### Resultado Obtido
![3sigmoid](https://imgur.com/1z5kkNu.png)

## Terceiro Teste
### Funções de Ativação
```python
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'tanh'))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))  
```

### Resultado Obtido
![3sigmoid](https://imgur.com/FhMDohH.png)

## Quarto Teste
### Funções de Ativação
```python
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'tanh', input_dim = 11))

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'tanh'))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))  
```
### Resultado Obtido
![3sigmoid](https://i.imgur.com/QYtNVNu.png)


## Consolidação dos Resultados
No código ann.py está descrito algumas informções adicionais acerca das funções de ativação.

![Results](https://i.imgur.com/BsHfwfb.png)

Na segunda parte do exercício o objetivo era tentar encontrar a acurácia, média e a variância. Para encontrar estes dados utilizei o *classificarion_report* e foi devolvido um dataframe de resultado

![Classification Metrics](https://imgur.com/FhMDohH.png)
Os resultados obtidos acima foram os melhores parâmetros que consegui obter através da mudança de parâmetros e das funções de ativação como visto no exemplo do Terceiro Teste. Note que o terceiro e o quarto teste ambos deram a mesma acurácia apesar da distinção de parâmetros entretanto o Terceiro Teste apresentou uma menor variância





