# Construindo uma Rede Neural Artificial

Construindo sua primeira rede neural usando Keras, Theano e Tensorflow
Parte da matéria de DeepLearning da escola de Engenharia de Software da UnB

### Autor
Davi de Alencar Mendes
16/0026415

## Funções de ativação
Referência: [Keras Layer Activation Functions](https://keras.io/api/layers/activations/)

### `softplus` function
`softplus(x) = log(exp(x) + 1)`

```python
>>> a = tf.constant([-20, -1.0, 0.0, 1.0, 20], dtype = tf.float32)
>>> b = tf.keras.activations.softplus(a)
>>> b.numpy()
array([2.0611537e-09, 3.1326166e-01, 6.9314718e-01, 1.3132616e+00,
         2.0000000e+01], dtype=float32)
```

### `tanh` function
Hyperbolic Tangent Function
```python
>>> a = tf.constant([-3.0,-1.0, 0.0,1.0,3.0], dtype = tf.float32)
>>> b = tf.keras.activations.tanh(a)
>>> b.numpy()
array([-0.9950547, -0.7615942,  0.,  0.7615942,  0.9950547], dtype=float32)
```

## Resultados

Originalmente, foi empregada a função de ativação `relu` para as 2 camadas. A proposta consiste em
utilizar `tanh` e `softplus` para substituir as funções de ativação originais.

| Metric    | Original | Proposto |
|-----------|----------|----------|
| Accuracy  | 86.40%   | 86.25%   |
| Precision | 88.21%   | 88.46%   |
| Recall    | 95.73%   | 95.17%   |
| F1 Score  | 91.82%   | 91.69%   |

O desempenho obtido é bastante próximo ao original.
