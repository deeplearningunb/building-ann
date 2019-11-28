# Construindo uma Rede Neural Artificial

Construindo sua primeira rede neural usando Keras, Theano e Tensorflow
Parte da matéria de DeepLearning da escola de Engenharia de Software da UnB

## Tarefas
* Crie uma branch com seu nome
* Troque a função de ativação
* Compile a rede
* Commit do resultado (NA SUA BRANCH)

## Funções de Ativação Utilizadas

Uma função utilizada para saber a saída do nó, ou seja, tenta determinar a saída da rede neural

> sigmoid: A principal razão pela qual usamos a função sigmóide é porque ela nos fornece uma forte precisão em contextos binários pois varia entre 0 e 1. Portanto, é especialmente usado em modelos onde temos que prever a probabilidade de existência de algo.

![Sigmoid Function Example](https://miro.medium.com/max/485/1*Xu7B5y9gp0iL5ooBj7LtWw.png)

> tanh: Bastante similar a função sigmoid descrita cima, entretanto consegue se dar bem com valores fortemente positivos ou fortemente negativos. 

![Tanh example](https://www.medcalc.org/manual/_help/functions/tanh.png)

## Resultados
No código ann.py está descrito algumas informções adicionais acerca das funções de ativação.

![Results](https://i.imgur.com/BsHfwfb.png)