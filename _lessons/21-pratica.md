---
layout: page
title: Aprendizado na Prática
nav_order: 21
---

[<img src="./colab_favicon_small.png" style="float: right;">](https://colab.research.google.com/github/icd-ufmg/icd-ufmg.github.io/blob/master/_lessons/21-pratica.ipynb)


# Aprendizado na Prática

{: .no_toc .mb-2 }

Fazendo uso a regressão e classicação knn e logística!
{: .fs-6 .fw-300 }

{: .no_toc .text-delta }
Resultados Esperados

1. Saber executar o KNN do SKlearn
1. Praticar o pipeline completo de ICD

---
**Sumário**
1. TOC
{:toc}
---


```python
#In: 
# -*- coding: utf8

from scipy import stats as ss

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['figure.figsize']  = (18, 10)
plt.rcParams['axes.labelsize']  = 20
plt.rcParams['axes.titlesize']  = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['lines.linewidth'] = 4
```


```python
#In: 
plt.ion()
plt.style.use('seaborn-colorblind')
plt.rcParams['figure.figsize']  = (12, 8)
```


```python
#In: 
def despine(ax=None):
    if ax is None:
        ax = plt.gca()
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
```


```python
#In: 
%%capture 
! wget https://github.com/icd-ufmg/material/raw/master/aulas/23-MLPratica/fashion/train-images-idx3-ubyte.gz -P fashion
! wget https://github.com/icd-ufmg/material/raw/master/aulas/23-MLPratica/fashion/t10k-images-idx3-ubyte.gz -P fashion
! wget https://github.com/icd-ufmg/material/raw/master/aulas/23-MLPratica/fashion/train-labels-idx1-ubyte.gz -P fashion
! wget https://github.com/icd-ufmg/material/raw/master/aulas/23-MLPratica/fashion/t10k-labels-idx1-ubyte.gz -P fashion
```


```python
#In: 
def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels
```

## Classificação

Nesta aula vamos explorar aprendizado de máquina na prática. Em particular, vamos iniciar por algoritmos de classificação na base Fashion MNIST. Depois disso vamos explorar regressão.

Acima, temos alguns códigos auxiliares para carregar a base. Nesta, cada ponto é um vetor de 784 posições. Ao redimensionar os mesmos com:

```python
#In: 
x.reshape((28, 28))
```

Temos uma imagem de alguma peça de vestimento. Código para carregar os dados abaixo. Vamos usar apenas 500 instâncias para treino e teste. Lento usar muito mais do que isso no meu computador.


```python
#In: 
X_train, y_train = load_mnist('fashion', kind='train')
X_test, y_test = load_mnist('fashion', kind='t10k')
```


```python
#In: 
X_train = X_train[:500]
y_train = y_train[:500]

X_test = X_test[:500]
y_test = y_test[:500]
```


```python
#In: 
np.unique(y_test, return_counts=True)
```




    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8),
     array([53, 40, 63, 46, 55, 52, 49, 52, 43, 47]))



Observe como cada instância é um vetor. Cada valor é um tom de cinza. 0 == branco; 256 == preto.


```python
#In: 
X_train[10]
```




    array([  0,   0,   0,   0,   1,   0,   0,   0,   0,  41, 162, 167,  84,
            30,  38,  94, 177, 176,  26,   0,   0,   0,   1,   0,   0,   0,
             0,   0,   0,   0,   0,   1,   0,   0,  41, 147, 228, 242, 228,
           236, 251, 251, 251, 255, 242, 230, 247, 221, 125,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,  91, 216, 228, 222,
           219, 219, 218, 222, 200, 224, 230, 221, 222, 222, 227, 237, 183,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   4, 202, 208,
           212, 217, 219, 222, 222, 219, 219, 220, 218, 222, 224, 224, 221,
           210, 227, 163,   0,   0,   0,   0,   0,   0,   0,   0,   0, 102,
           225, 210, 216, 218, 222, 221, 219, 225, 225, 221, 222, 224, 222,
           224, 224, 215, 215, 218,  28,   0,   0,   0,   0,   0,   0,   0,
             0, 189, 222, 220, 213, 219, 220, 218, 221, 220, 219, 222, 226,
           222, 220, 221, 216, 215, 218, 229, 148,   0,   0,   0,   0,   0,
             0,   0,  11, 240, 210, 227, 213, 214, 220, 217, 220, 224, 220,
           221, 217, 206, 209, 208, 212, 220, 224, 218, 234,   0,   0,   0,
             0,   0,   0,   0, 118, 214, 208, 224, 216, 211, 226, 212, 219,
           213, 193, 192, 179, 194, 213, 216, 216, 217, 227, 216, 221,  91,
             0,   0,   0,   0,   0,   0, 170, 221, 205, 225, 219, 217, 232,
           232, 226, 182, 182, 211, 226, 220, 212, 217, 216, 216, 225, 213,
           226, 184,   0,   0,   0,   0,   0,   0,   0, 181, 229, 219, 220,
           213, 227, 226, 222, 214, 222, 220, 216, 215, 213, 214, 216, 215,
           220, 233, 211,   0,   0,   0,   0,   0,   0,   0,   0,   0, 164,
           242, 222, 210, 214, 211, 215, 215, 216, 217, 215, 215, 215, 215,
           213, 222, 238, 184,   0,   0,   0,   0,   0,   0,   0,   0,   2,
             0,   0,  60, 222, 217, 214, 214, 215, 219, 202, 217, 210, 203,
           216, 212, 221, 200,  60,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0, 193, 222, 208, 216, 215, 216, 218, 220,
           219, 215, 216, 204, 222, 148,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0, 194, 222, 206, 216, 216, 217,
           218, 217, 218, 216, 218, 208, 219, 179,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   1,   0,   0, 192, 224, 213, 217,
           217, 218, 217, 217, 217, 215, 216, 209, 215, 176,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0, 194, 221,
           214, 217, 216, 216, 217, 217, 217, 216, 214, 210, 214, 177,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,
           193, 220, 214, 218, 217, 216, 217, 217, 216, 216, 215, 212, 214,
           183,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,
             0,   0, 197, 220, 214, 219, 218, 218, 218, 218, 217, 217, 219,
           214, 217, 189,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0, 201, 222, 214, 219, 218, 219, 219, 218, 218,
           217, 219, 216, 220, 196,   0,   0,   1,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0, 209, 222, 216, 220, 219, 219, 220,
           220, 218, 217, 219, 216, 222, 203,   0,   0,   1,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0, 209, 221, 216, 220, 219,
           219, 221, 221, 219, 219, 221, 217, 222, 210,   0,   0,   1,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 208, 222, 218,
           221, 220, 220, 221, 222, 220, 220, 222, 219, 222, 216,   0,   0,
             1,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0, 210,
           226, 220, 221, 220, 221, 222, 222, 220, 220, 224, 221, 224, 221,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,
             0, 217, 227, 219, 222, 224, 219, 219, 221, 222, 220, 221, 222,
           225, 220,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             1,   0,   0, 183, 228, 221, 225, 221, 215, 217, 221, 222, 221,
           222, 224, 224, 193,   0,   0,   1,   0,   0,   0,   0,   0,   0,
             0,   0,   1,   0,   0, 179, 225, 218, 221, 219, 213, 213, 217,
           220, 219, 218, 221, 222, 197,   0,   0,   2,   0,   0,   0,   0,
             0,   0,   0,   0,   1,   0,   0, 240, 233, 228, 235, 232, 229,
           228, 229, 231, 231, 231, 228, 229, 212,   0,   0,   1,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0, 101, 157, 148, 148,
           167, 180, 182, 179, 176, 172, 171, 164, 177, 163,   0,   0,   1,
             0,   0,   0,   0], dtype=uint8)



Ao redimensionar temos uma peça de roupa! Fashion!


```python
#In: 
I = X_train[0].reshape(28, 28)
print(I.shape)
```

    (28, 28)



```python
#In: 
plt.imshow(X_train[100].reshape(28, 28))
```




    <matplotlib.image.AxesImage at 0x7f8cda558fa0>




    
![png](21-pratica_files/21-pratica_14_1.png)
    



```python
#In: 
plt.imshow(X_train[1].reshape(28, 28))
```




    <matplotlib.image.AxesImage at 0x7f8cd8252110>




    
![png](21-pratica_files/21-pratica_15_1.png)
    



```python
#In: 
M = np.array([[1, 2], [2, 3]])
M.flatten()
```




    array([1, 2, 2, 3])



Temos 10 classes. 


```python
#In: 
len(set(y_train))
```




    10




```python
#In: 
text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
index = np.arange(len(text_labels))
labels = pd.Series(text_labels, index=index)
labels
```




    0       t-shirt
    1       trouser
    2      pullover
    3         dress
    4          coat
    5        sandal
    6         shirt
    7       sneaker
    8           bag
    9    ankle boot
    dtype: object



## Executando o Scikit-Learn

Agora, vamos executar o código do sklearn na nossa base. Lembrando que temos que separar a mesma em Treino, Validação e Teste. Para tal, vamos fazer uso da classe `StratifiedKFold`. A mesma serve para realizar n-fold cross validation. A biblioteca sklearn não cria grupos de validação para você, a mesma só usa o conceito de treino/teste. De qualquer forma, validação nada mais é do que um conjunto a mais de teste. Então, vamos fazer 5-fold no nosso treino, separando em treino/validação. Note que NUNCA avaliamos nada no teste, apenas reportamos os números no fim!!


```python
#In: 
from sklearn.model_selection import StratifiedKFold
```

Ao gerar o split, teremos 20 conjuntos (muito eu sei).


```python
#In: 
skf = StratifiedKFold(n_splits=20, shuffle=True)
```

Cada passo do laço retorna indices do vetor


```python
#In: 
for treino, validacao in skf.split(X_train, y_train):
    count_train = np.unique(y_train[treino], return_counts=True)
    print(count_train)
```

    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([49, 37, 42, 62, 48, 39, 58, 49, 43, 48]))
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([49, 37, 42, 62, 48, 39, 58, 49, 43, 48]))
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([49, 36, 42, 62, 48, 39, 58, 50, 43, 48]))
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([49, 36, 42, 62, 48, 39, 58, 50, 43, 48]))
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([49, 36, 42, 62, 49, 38, 58, 50, 43, 48]))
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([49, 36, 43, 62, 49, 39, 58, 50, 42, 47]))
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([49, 36, 43, 62, 49, 39, 58, 50, 42, 47]))
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([49, 36, 43, 62, 49, 39, 58, 50, 42, 47]))
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([50, 36, 43, 61, 49, 39, 58, 50, 42, 47]))
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([50, 36, 43, 61, 49, 39, 58, 50, 42, 47]))
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([50, 36, 43, 61, 49, 39, 58, 49, 43, 47]))
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([50, 36, 43, 61, 49, 39, 58, 49, 43, 47]))
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([50, 36, 43, 61, 49, 39, 58, 49, 43, 47]))
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([50, 36, 43, 62, 48, 39, 58, 49, 43, 47]))
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([50, 36, 43, 62, 48, 39, 58, 49, 43, 47]))
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([50, 36, 43, 62, 48, 39, 57, 49, 43, 48]))
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([49, 36, 43, 62, 48, 39, 58, 49, 43, 48]))
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([49, 36, 43, 62, 48, 39, 58, 49, 43, 48]))
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([49, 36, 43, 62, 48, 39, 58, 49, 43, 48]))
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([49, 36, 43, 62, 48, 39, 58, 49, 43, 48]))


Vamos quebrar nos conjuntos e avaliar o KNN. De um mundo de métricas, vamos fazer uso de 4 neste notebook:

1. Precisão
2. Revocação
3. F1
4. Acurácia

![](https://raw.githubusercontent.com/icd-ufmg/material/master/aulas/23-MLPratica/f.png)

Na figura acima, assuma que o termo `busca` indica as previsões do seu classificador (sem tempo para alterar a figura irmão). Sendo `y_p (y-pred)` um conjunto de elementos da previsão e `y_t (y-true)` os rótulos reais. Por clareza, vamos assumir duas classes `1 e 0`. Afinal, o caso multiclasse pode ser reduzido para este. Assim, cada elemento dos vetores `y_p` e `y_t` $\in \{0, 1\}$. Os verdadeiros positivos, __true positive (TP)__, é o conjunto de previsões da classe `1` que foram corretas. Podemos formalizar como:

$$TP = \sum_i \mathbb{1}_{y_t[i] = 1} \mathbb{1}_{y_p[i] = 1}$$

$\mathbb{1}_{y_t[i] = 1}$ retorna 1 quando $y_t[i] = 1$, 0 caso contrário. O mesmo vale para $\mathbb{1}_{y_t[i] = y_p[i]}$ que retorna um quando $y_p[i] = 1$. Usando a mesma notação, os verdadeiros negativos é definido como:

$$TN = \sum_i \mathbb{1}_{y_t[i] = 0} \mathbb{1}_{y_t[i] = 0}$$

Os falsos positivos e negativos capturam os erros da previsão. Note que nos dois a previsão é o oposto do real:

$$FP = \sum_i \mathbb{1}_{y_t[i] = 0} \mathbb{1}_{y_p[i] = 1}$$

$$FN = \sum_i \mathbb{1}_{y_t[i] = 1} \mathbb{1}_{y_p[i] = 0}$$

Assim, a acurácia do classificador é definida como a fração total de acertos:

$$Acuracia = \frac{TP + TN}{TP + TN + FP + FN}$$

A precisão é definida como a fração dos elementos classificados como 1 que foram corretos:

$$Precisão = \frac{TP}{TP + FP}$$

A revocação é a fração de todos os elementos do conjunto 1 que foram acertados. Diferente da precisão, aqui focamos nos elementos reais! Na precisão focamos nas previsões.

$$Revocação = \frac{TP}{TP + FN}$$

Tanto a previsão quanto a revocação importam. Na primeira, precisão, queremos saber o quão bom o classificador é em retornar acertos. Na segunda, o quanto de elementos reais o classificador captura. Observe como um classificador que sempre retorna 1 tem revocação máxima, porém precisão baixa. Um classificador que sempre retorna 0 tem precisão máxima e revocação baixa. Para captura a média harmônica dos dois usamos o F1-score:

$$F1 = MediaHarmonica(Precisao, Revocacao)$$

Dependendo do problema uma métrica pode importar mais do que a outra. Aqui, trabalhamos com classes balanceadas, então a acurácia já é boa suficiente. Vamos avaliar a acurácia nos conjuntos abaixo:


```python
#In: 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
```

Observe como o laço abaixo guarda o melhor valor de n para cada fold de validação!


```python
#In: 
fold = 0
melhores = []
for treino, validacao in skf.split(X_train, y_train):
    X_tt = X_train[treino]
    y_tt = y_train[treino]
    X_v = X_train[validacao]
    y_v = y_train[validacao]
    
    best = (0, 0)
    for nn in [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100]: # Vamos testar tais valores de n
        model = KNeighborsClassifier(n_neighbors=nn)
        model.fit(X_tt, y_tt) # treina no conjunto de treino
        y_pv = model.predict(X_v) # previsões no conjunto de validação
        
        # Resultado com melhor acurácia!
        accuracy = accuracy_score(y_v, y_pv)
        if accuracy > best[0]:
            best = (accuracy, nn)
    
    melhores.append(best[1])
    fold += 1
    print('Fold-{}, melhor nn = {}, acc = {}'.format(fold, best[1], best[0]))
```

    Fold-1, melhor nn = 2, acc = 0.84
    Fold-2, melhor nn = 4, acc = 0.84
    Fold-3, melhor nn = 4, acc = 0.68
    Fold-4, melhor nn = 2, acc = 0.68
    Fold-5, melhor nn = 30, acc = 0.6
    Fold-6, melhor nn = 5, acc = 0.8
    Fold-7, melhor nn = 2, acc = 0.76
    Fold-8, melhor nn = 5, acc = 0.92
    Fold-9, melhor nn = 2, acc = 0.72
    Fold-10, melhor nn = 9, acc = 0.68
    Fold-11, melhor nn = 3, acc = 0.68
    Fold-12, melhor nn = 2, acc = 0.68
    Fold-13, melhor nn = 3, acc = 0.76
    Fold-14, melhor nn = 8, acc = 0.88
    Fold-15, melhor nn = 30, acc = 0.84
    Fold-16, melhor nn = 8, acc = 0.84
    Fold-17, melhor nn = 40, acc = 0.76
    Fold-18, melhor nn = 7, acc = 0.92
    Fold-19, melhor nn = 9, acc = 0.68
    Fold-20, melhor nn = 20, acc = 0.88


Vamos ver quantas vezes cada escolha de número de vizinhos, nn, ganhou na validação.


```python
#In: 
unique, counts = np.unique(melhores, return_counts=True)
plt.bar(unique, counts)
despine()
plt.title('Número de vezes que n ganhou na validação')
plt.xlabel('NN')
plt.ylabel('Count na validação')
```




    Text(0, 0.5, 'Count na validação')




    
![png](21-pratica_files/21-pratica_31_1.png)
    


Agora, podemos finalmente avaliar o modelo no conjunto de teste! Vamos escolher n como a médiana dos folds.


```python
#In: 
print(np.median(melhores))
```

    5.0


Vamos verificar as outras métricas e todas as classes.


```python
#In: 
from sklearn.metrics import classification_report

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))
```

                  precision    recall  f1-score   support
    
               0       0.71      0.77      0.74        53
               1       0.95      0.97      0.96        40
               2       0.53      0.62      0.57        63
               3       0.86      0.93      0.90        46
               4       0.63      0.60      0.62        55
               5       0.94      0.60      0.73        52
               6       0.53      0.43      0.47        49
               7       0.71      0.87      0.78        52
               8       0.94      0.79      0.86        43
               9       0.83      0.96      0.89        47
    
        accuracy                           0.74       500
       macro avg       0.76      0.75      0.75       500
    weighted avg       0.75      0.74      0.74       500
    


Parece que erramos muito a classe 4, coat. Casacos se parecem com camisas, vestidos etc. Podemos investigar isto usando a matriz de confusão.


```python
#In: 
from sklearn.metrics import confusion_matrix
plt.imshow(confusion_matrix(y_test, model.predict(X_test)))
plt.xticks(labels.index, labels, rotation=90)
plt.yticks(labels.index, labels);
```


    
![png](21-pratica_files/21-pratica_37_0.png)
    


## Logística

Vamos repetir tudo para a regressão logística. Felizmente, o sklearn tem uma versão da logística que já faz treino/validação internamente. Para alguns modelos, existem atalhos para fazer isto. Caso queira entender, leia:

https://robjhyndman.com/hyndsight/crossvalidation/


```python
#In: 
from sklearn.linear_model import LogisticRegressionCV
```


```python
#In: 
# O LogisticCV tenta várias regularizações.
model = LogisticRegressionCV(Cs=100,
                             penalty='l2',   #ridge
                             cv=5,           #5 folds internos
                             fit_intercept=False,
                             solver='liblinear',
                             multi_class='ovr')
model.fit(X_train, y_train)
```




    LogisticRegressionCV(Cs=100, cv=5, fit_intercept=False, multi_class='ovr',
                         solver='liblinear')




```python
#In: 
model.C_
```




    array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
           0.0001, 0.0001])




```python
#In: 
print(classification_report(y_test, model.predict(X_test)))
```

                  precision    recall  f1-score   support
    
               0       0.83      0.64      0.72        53
               1       1.00      0.97      0.99        40
               2       0.69      0.54      0.61        63
               3       0.75      0.91      0.82        46
               4       0.59      0.75      0.66        55
               5       1.00      0.69      0.82        52
               6       0.52      0.65      0.58        49
               7       0.85      0.87      0.86        52
               8       0.89      0.79      0.84        43
               9       0.82      0.98      0.89        47
    
        accuracy                           0.77       500
       macro avg       0.79      0.78      0.78       500
    weighted avg       0.79      0.77      0.77       500
    



```python
#In: 
plt.imshow(confusion_matrix(y_test, model.predict(X_test)))
plt.xticks(labels.index, labels, rotation=90)
plt.yticks(labels.index, labels);
```


    
![png](21-pratica_files/21-pratica_43_0.png)
    


## Regressão

Agora vamos avaliar modelos de regressão em dados tabulares. Primeiro, vamos carregar os dados. Obsevre que cada atributo é diferente. Data, numéricos categóricos, etc...


```python
#In: 
df = pd.read_csv('walmart.csv', on_bad_lines='skip')
df = df.iloc[:, :-1]
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>Date</th>
      <th>Weekly_Sales_Store</th>
      <th>IsHoliday</th>
      <th>Temperature</th>
      <th>Fuel_Price</th>
      <th>MarkDown1</th>
      <th>MarkDown2</th>
      <th>MarkDown3</th>
      <th>MarkDown4</th>
      <th>MarkDown5</th>
      <th>CPI</th>
      <th>Unemployment</th>
      <th>Type</th>
      <th>Size</th>
      <th>Total Markdown</th>
      <th>Total Sales</th>
      <th>SalesPerSqFt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5/2/2010</td>
      <td>1643690.90</td>
      <td>False</td>
      <td>42.31</td>
      <td>2.572</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>211.096358</td>
      <td>8.106</td>
      <td>A</td>
      <td>151315</td>
      <td>0.0</td>
      <td>1643690.90</td>
      <td>10.862710</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>12/2/2010</td>
      <td>1641957.44</td>
      <td>True</td>
      <td>38.51</td>
      <td>2.548</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>211.242170</td>
      <td>8.106</td>
      <td>A</td>
      <td>151315</td>
      <td>0.0</td>
      <td>1641957.44</td>
      <td>10.851254</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>19/2/2010</td>
      <td>1611968.17</td>
      <td>False</td>
      <td>39.93</td>
      <td>2.514</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>211.289143</td>
      <td>8.106</td>
      <td>A</td>
      <td>151315</td>
      <td>0.0</td>
      <td>1611968.17</td>
      <td>10.653063</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>26/2/2010</td>
      <td>1409727.59</td>
      <td>False</td>
      <td>46.63</td>
      <td>2.561</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>211.319643</td>
      <td>8.106</td>
      <td>A</td>
      <td>151315</td>
      <td>0.0</td>
      <td>1409727.59</td>
      <td>9.316509</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5/3/2010</td>
      <td>1554806.68</td>
      <td>False</td>
      <td>46.50</td>
      <td>2.625</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>211.350143</td>
      <td>8.106</td>
      <td>A</td>
      <td>151315</td>
      <td>0.0</td>
      <td>1554806.68</td>
      <td>10.275298</td>
    </tr>
  </tbody>
</table>
</div>



Vamos criar o treino e teste


```python
#In: 
from sklearn.model_selection import train_test_split
```


```python
#In: 
train_df, test_df = train_test_split(df, test_size=0.2)
```


```python
#In: 
df.shape
```




    (6435, 18)




```python
#In: 
train_df.shape
```




    (5148, 18)




```python
#In: 
test_df.shape
```




    (1287, 18)



Segundo, temos que converter os atributos categóricos em colunas novas. Para isto, fazemos uso de one hot encoding. Cada categoria vira uma coluna de 1/0. Algoritmos como KNN e Logistic não sabem fazer uso de categorias por padrão. Mesmo se as categorias representarem números, faça uso de one hot. Sempre se pergunte: faz sentido computar uma distância nessa coluna? Se não, one-hot (ou outra abordagem).


```python
#In: 
train_df = train_df.drop(['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], axis='columns')
test_df = test_df.drop(['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], axis='columns')
```


```python
#In: 
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>Date</th>
      <th>Weekly_Sales_Store</th>
      <th>IsHoliday</th>
      <th>Temperature</th>
      <th>Fuel_Price</th>
      <th>CPI</th>
      <th>Unemployment</th>
      <th>Type</th>
      <th>Size</th>
      <th>Total Markdown</th>
      <th>Total Sales</th>
      <th>SalesPerSqFt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5798</th>
      <td>41</td>
      <td>5/8/2011</td>
      <td>1402233.69</td>
      <td>False</td>
      <td>72.19</td>
      <td>3.554</td>
      <td>193.911013</td>
      <td>6.901</td>
      <td>A</td>
      <td>196321</td>
      <td>0.00</td>
      <td>1402233.69</td>
      <td>7.142556</td>
    </tr>
    <tr>
      <th>5518</th>
      <td>39</td>
      <td>16/9/2011</td>
      <td>1372500.63</td>
      <td>False</td>
      <td>83.11</td>
      <td>3.526</td>
      <td>214.793411</td>
      <td>8.177</td>
      <td>A</td>
      <td>184109</td>
      <td>0.00</td>
      <td>1372500.63</td>
      <td>7.454826</td>
    </tr>
    <tr>
      <th>1608</th>
      <td>12</td>
      <td>8/10/2010</td>
      <td>918335.68</td>
      <td>False</td>
      <td>71.82</td>
      <td>3.013</td>
      <td>126.279167</td>
      <td>14.313</td>
      <td>B</td>
      <td>112238</td>
      <td>0.00</td>
      <td>918335.68</td>
      <td>8.182039</td>
    </tr>
    <tr>
      <th>848</th>
      <td>6</td>
      <td>24/8/2012</td>
      <td>1501095.49</td>
      <td>False</td>
      <td>79.03</td>
      <td>3.620</td>
      <td>223.786018</td>
      <td>5.668</td>
      <td>A</td>
      <td>202505</td>
      <td>17191.42</td>
      <td>1518286.91</td>
      <td>7.497528</td>
    </tr>
    <tr>
      <th>1756</th>
      <td>13</td>
      <td>12/11/2010</td>
      <td>1939964.63</td>
      <td>False</td>
      <td>42.55</td>
      <td>2.831</td>
      <td>126.546161</td>
      <td>7.795</td>
      <td>A</td>
      <td>219622</td>
      <td>0.00</td>
      <td>1939964.63</td>
      <td>8.833198</td>
    </tr>
  </tbody>
</table>
</div>




```python
#In: 
train_df.dtypes
```




    Store                   int64
    Date                   object
    Weekly_Sales_Store    float64
    IsHoliday                bool
    Temperature           float64
    Fuel_Price            float64
    CPI                   float64
    Unemployment          float64
    Type                   object
    Size                    int64
    Total Markdown        float64
    Total Sales           float64
    SalesPerSqFt          float64
    dtype: object



Vamos inicialmente converter a data. Note que a mesma existe em uma escala completamente diferente do resto. O split abaixo quebra o texto da data.


```python
#In: 
train_df['Date'].str.split('/')
```




    5798      [5, 8, 2011]
    5518     [16, 9, 2011]
    1608     [8, 10, 2010]
    848      [24, 8, 2012]
    1756    [12, 11, 2010]
                 ...      
    449      [25, 6, 2010]
    1783     [20, 5, 2011]
    3804     [30, 9, 2011]
    5437     [26, 2, 2010]
    5714     [21, 9, 2012]
    Name: Date, Length: 5148, dtype: object



Agora pegamos o mês


```python
#In: 
for split in train_df['Date'].str.split('/'):
    print(split[1])
    break
```

    8



```python
#In: 
train_df['Month'] = [split[1] for split in train_df['Date'].str.split('/')]
test_df['Month'] = [split[1] for split in test_df['Date'].str.split('/')]
```


```python
#In: 
test_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>Date</th>
      <th>Weekly_Sales_Store</th>
      <th>IsHoliday</th>
      <th>Temperature</th>
      <th>Fuel_Price</th>
      <th>CPI</th>
      <th>Unemployment</th>
      <th>Type</th>
      <th>Size</th>
      <th>Total Markdown</th>
      <th>Total Sales</th>
      <th>SalesPerSqFt</th>
      <th>Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2765</th>
      <td>20</td>
      <td>7/1/2011</td>
      <td>1843030.95</td>
      <td>False</td>
      <td>31.43</td>
      <td>3.193</td>
      <td>204.648780</td>
      <td>7.343</td>
      <td>A</td>
      <td>203742</td>
      <td>0.00</td>
      <td>1843030.95</td>
      <td>9.045906</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4573</th>
      <td>32</td>
      <td>12/10/2012</td>
      <td>1176681.31</td>
      <td>False</td>
      <td>43.49</td>
      <td>3.760</td>
      <td>199.053937</td>
      <td>7.557</td>
      <td>A</td>
      <td>203007</td>
      <td>8937.26</td>
      <td>1185618.57</td>
      <td>5.840284</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4736</th>
      <td>34</td>
      <td>4/6/2010</td>
      <td>966187.51</td>
      <td>False</td>
      <td>72.17</td>
      <td>2.701</td>
      <td>126.136065</td>
      <td>9.593</td>
      <td>A</td>
      <td>158114</td>
      <td>0.00</td>
      <td>966187.51</td>
      <td>6.110702</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3667</th>
      <td>26</td>
      <td>11/11/2011</td>
      <td>1077640.13</td>
      <td>False</td>
      <td>40.08</td>
      <td>3.570</td>
      <td>136.461806</td>
      <td>7.598</td>
      <td>A</td>
      <td>152513</td>
      <td>18336.76</td>
      <td>1095976.89</td>
      <td>7.186121</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2503</th>
      <td>18</td>
      <td>24/6/2011</td>
      <td>920719.98</td>
      <td>False</td>
      <td>67.41</td>
      <td>3.851</td>
      <td>135.265267</td>
      <td>8.975</td>
      <td>B</td>
      <td>120653</td>
      <td>0.00</td>
      <td>920719.98</td>
      <td>7.631140</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



Removendo a data


```python
#In: 
train_df = train_df.drop(['Date'], axis='columns')
test_df = test_df.drop(['Date'], axis='columns')
train_df.shape
```




    (5148, 13)



One hot encoding do resto categórico


```python
#In: 
cols_usar = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size',
             'Weekly_Sales_Store',
             'Store', 'Month', 'Type', 'IsHoliday']
cols_cat = ['Store', 'Month', 'Type', 'IsHoliday']
train_df = pd.get_dummies(train_df[cols_usar],
                          columns=cols_cat)
test_df = pd.get_dummies(test_df[cols_usar],
                         columns=cols_cat)
```

Vamos focar em poucos pontos para o notebook executar e só olhar para o teste no fim de tudo!


```python
#In: 
train_df = train_df.sample(1000)
test_df = test_df.sample(1000)
```

*Normalizando dados*: Como trabalhar com esse mundo de valores distintos? Solução!? Normalizar!


```python
#In: 
y_train_df = train_df['Weekly_Sales_Store']
X_train_df = train_df.drop('Weekly_Sales_Store', axis='columns')
```


```python
#In: 
X_train_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Temperature</th>
      <th>Fuel_Price</th>
      <th>CPI</th>
      <th>Unemployment</th>
      <th>Size</th>
      <th>Store_1</th>
      <th>Store_2</th>
      <th>Store_3</th>
      <th>Store_4</th>
      <th>Store_5</th>
      <th>...</th>
      <th>Month_5</th>
      <th>Month_6</th>
      <th>Month_7</th>
      <th>Month_8</th>
      <th>Month_9</th>
      <th>Type_A</th>
      <th>Type_B</th>
      <th>Type_C</th>
      <th>IsHoliday_False</th>
      <th>IsHoliday_True</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>525</th>
      <td>31.64</td>
      <td>3.153</td>
      <td>129.855533</td>
      <td>5.143</td>
      <td>205863</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5266</th>
      <td>76.55</td>
      <td>3.688</td>
      <td>220.407558</td>
      <td>6.989</td>
      <td>39910</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4853</th>
      <td>75.89</td>
      <td>3.646</td>
      <td>130.885355</td>
      <td>9.285</td>
      <td>158114</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4439</th>
      <td>42.43</td>
      <td>2.692</td>
      <td>189.734262</td>
      <td>9.014</td>
      <td>203007</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1471</th>
      <td>58.63</td>
      <td>2.771</td>
      <td>215.207452</td>
      <td>7.564</td>
      <td>207499</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3575</th>
      <td>9.55</td>
      <td>2.788</td>
      <td>131.527903</td>
      <td>8.488</td>
      <td>152513</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1299</th>
      <td>69.76</td>
      <td>3.105</td>
      <td>126.380567</td>
      <td>9.524</td>
      <td>126512</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2770</th>
      <td>25.38</td>
      <td>3.239</td>
      <td>206.076386</td>
      <td>7.343</td>
      <td>203742</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3392</th>
      <td>31.92</td>
      <td>3.737</td>
      <td>136.959839</td>
      <td>8.659</td>
      <td>203819</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>788</th>
      <td>87.00</td>
      <td>3.524</td>
      <td>216.725739</td>
      <td>6.925</td>
      <td>202505</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 67 columns</p>
</div>




```python
#In: 
from sklearn.preprocessing import StandardScaler
```

**IMPORTANTE SÓ NORMALIZE O TREINO!!! DEPOIS USE A MÉDIA E DESVIO DO TREINO PARA NORMALIZAR O TESTE!!**

**O TESTE É UM FUTURO! NÃO EXISTE, VOCÊ NÃO SABE NADA DO MESMO**


```python
#In: 
scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_x.fit_transform(X_train_df.values)
y_train = scaler_y.fit_transform(y_train_df.values[:, np.newaxis])
```


```python
#In: 
X_train
```




    array([[-1.47899696, -0.37784933, -1.08529399, ..., -0.39166636,
             0.29287596, -0.29287596],
           [ 0.89990939,  0.797513  ,  1.21462889, ...,  2.55319349,
             0.29287596, -0.29287596],
           [ 0.86494884,  0.70524157, -1.05913764, ..., -0.39166636,
             0.29287596, -0.29287596],
           ...,
           [-1.81059246, -0.18891258,  0.85063272, ..., -0.39166636,
            -3.41441472,  3.41441472],
           [-1.46416522,  0.90516301, -0.90485236, ..., -0.39166636,
             0.29287596, -0.29287596],
           [ 1.4534514 ,  0.43721502,  1.12111469, ..., -0.39166636,
             0.29287596, -0.29287596]])




```python
#In: 
y_train
```




    array([[ 2.62453291e+00],
           [-9.11982774e-01],
           [-1.88058619e-01],
           [ 3.74069171e-02],
           [ 4.52591455e-01],
           [-9.77094958e-01],
           [ 4.02472469e-01],
           [-3.61349552e-01],
           [ 8.28706692e-02],
           [-3.48086075e-01],
           [ 9.55379822e-01],
           [-2.75153083e-01],
           [ 5.33213308e-01],
           [-8.13621395e-01],
           [-6.36005303e-02],
           [ 1.54425972e-01],
           [ 1.72205727e+00],
           [-1.42171707e+00],
           [ 1.62484616e+00],
           [-1.19794621e+00],
           [ 3.90878783e-01],
           [-2.88423915e-01],
           [-9.43327418e-01],
           [ 8.70980877e-02],
           [ 1.40786373e+00],
           [-1.04198434e+00],
           [ 6.06612877e-03],
           [-1.04759308e+00],
           [-1.31008988e+00],
           [-1.03809762e+00],
           [-2.34465051e-01],
           [-2.39539942e-01],
           [-1.09658018e+00],
           [-9.67785169e-01],
           [ 5.12377672e-01],
           [ 1.84777424e+00],
           [-9.82714848e-01],
           [-8.50257605e-01],
           [ 6.30126891e-01],
           [ 1.73396906e+00],
           [-9.03445727e-01],
           [-6.44217310e-01],
           [ 1.77278823e+00],
           [ 6.60681673e-01],
           [-5.46228154e-01],
           [-2.58561896e-01],
           [-8.50122998e-01],
           [-4.82123830e-02],
           [-3.99420526e-01],
           [ 8.09129939e-01],
           [-7.56418995e-01],
           [-2.15915400e-01],
           [ 1.91070846e+00],
           [-8.69161841e-01],
           [-5.69915876e-01],
           [-1.98333921e-01],
           [ 2.12706837e-01],
           [ 4.12762624e-01],
           [-2.83251336e-01],
           [-1.13757894e+00],
           [-1.05822086e-01],
           [-1.30333045e+00],
           [-1.08285638e+00],
           [ 1.07209265e+00],
           [-1.25428881e+00],
           [-2.88224093e-01],
           [ 7.09604225e-01],
           [-6.74222576e-01],
           [-3.60816461e-01],
           [ 1.55103688e+00],
           [-9.23001808e-01],
           [ 6.39292558e-01],
           [ 9.08511441e-01],
           [ 7.15383607e-02],
           [ 2.90754888e-01],
           [-6.36226975e-01],
           [-2.11907352e-01],
           [ 1.60141463e+00],
           [ 2.76556364e+00],
           [ 7.86259216e-01],
           [-2.93448319e-01],
           [ 8.80635987e-01],
           [-8.02218373e-01],
           [-8.82012730e-01],
           [ 6.59089611e-01],
           [-2.73535439e-01],
           [ 1.64418010e-01],
           [-7.11721416e-01],
           [-2.70289725e-01],
           [-8.08215395e-01],
           [-1.22081147e+00],
           [-1.52339899e-01],
           [ 4.22690929e-01],
           [-8.06615532e-01],
           [-9.02737717e-01],
           [-7.93101550e-01],
           [-4.80053127e-02],
           [-7.71355191e-01],
           [-7.97497942e-01],
           [ 4.93670634e-01],
           [-1.13281584e+00],
           [-1.29291035e+00],
           [ 1.43199093e+00],
           [-1.05292993e-01],
           [ 1.69287578e+00],
           [ 1.01242457e-01],
           [-2.53061047e-01],
           [-6.89994140e-01],
           [-8.55166666e-02],
           [-1.41489604e+00],
           [ 4.61323308e-01],
           [-2.04685136e-01],
           [ 7.14465619e-01],
           [-9.36737767e-01],
           [-1.01590029e+00],
           [ 1.79002518e+00],
           [ 1.62519326e+00],
           [ 1.61673437e+00],
           [ 1.32396257e+00],
           [-1.01987622e+00],
           [ 2.47633374e-01],
           [-3.87987083e-01],
           [-1.11600341e+00],
           [ 2.33770001e-02],
           [-3.95151333e-01],
           [-9.70825312e-01],
           [ 1.66122586e+00],
           [ 5.56820587e-01],
           [ 6.48766607e-01],
           [ 1.87434355e-01],
           [-1.20365553e+00],
           [-3.88648783e-01],
           [-8.24455514e-01],
           [-6.83777176e-01],
           [ 7.17416321e-01],
           [ 8.24844804e-01],
           [ 1.73199978e+00],
           [ 1.67322521e+00],
           [-2.57008985e-01],
           [ 7.10625794e-01],
           [-9.45378983e-01],
           [ 1.86032342e+00],
           [-7.07337128e-01],
           [ 1.95868666e+00],
           [-2.24889260e-01],
           [ 1.89532305e+00],
           [ 1.28739921e-01],
           [-9.93268130e-01],
           [ 7.59860746e-01],
           [-3.66687284e-01],
           [ 4.20664982e-01],
           [-4.84675565e-01],
           [ 2.93300322e-01],
           [ 3.99730424e-02],
           [ 5.05068264e-01],
           [-1.30509923e+00],
           [ 7.35228362e-02],
           [-1.08252900e+00],
           [-9.89918390e-01],
           [ 1.18883696e+00],
           [-1.29918164e-01],
           [-1.10102112e+00],
           [ 1.56847082e-01],
           [-9.44189405e-01],
           [-8.90473726e-01],
           [ 1.10572794e-01],
           [-9.52891284e-01],
           [ 1.62145527e+00],
           [ 1.50303846e+00],
           [ 2.57312548e+00],
           [ 1.71428098e+00],
           [ 1.16952469e+00],
           [ 1.72861009e+00],
           [-3.51472216e-01],
           [-1.27655833e+00],
           [ 4.37185241e-01],
           [-2.41404663e-01],
           [ 8.71529786e-01],
           [ 1.95704289e+00],
           [-2.75112933e-01],
           [-1.31700853e-01],
           [ 1.50945221e+00],
           [-1.02381863e+00],
           [ 2.38943813e-01],
           [ 2.43849642e-01],
           [-1.24541003e+00],
           [-7.69684097e-01],
           [-4.26102617e-01],
           [-1.82665902e-01],
           [ 1.24254027e+00],
           [-9.65225220e-01],
           [ 1.29484478e+00],
           [-2.33190157e-02],
           [ 1.54529404e+00],
           [-1.35460561e+00],
           [-8.93755396e-02],
           [-1.33888304e+00],
           [ 6.93270149e-01],
           [ 9.09055119e-01],
           [ 4.79736691e-01],
           [-2.05362385e-01],
           [-2.64513270e-01],
           [-1.37234814e+00],
           [-9.77257469e-01],
           [-5.63676329e-01],
           [ 1.62601793e-01],
           [-1.60427922e-01],
           [-8.46168490e-01],
           [ 5.58714229e-01],
           [ 2.09186374e-01],
           [-9.46645042e-01],
           [ 3.12691055e-01],
           [-1.33366042e+00],
           [ 1.49966445e+00],
           [ 1.85384553e+00],
           [-1.12776649e+00],
           [-7.56743392e-01],
           [ 6.85984770e-02],
           [ 5.16393698e-02],
           [-3.07823859e-01],
           [-1.26688740e+00],
           [-4.90703436e-01],
           [ 1.00423539e+00],
           [ 1.73083222e+00],
           [ 1.02954279e+00],
           [-9.28683039e-01],
           [-1.09877607e+00],
           [-1.29342926e+00],
           [ 1.87707637e+00],
           [-8.79599672e-01],
           [-1.21310389e+00],
           [-9.35639807e-01],
           [-2.15179880e-01],
           [-4.12443923e-01],
           [-2.41363010e-03],
           [-7.38259171e-02],
           [-9.94255780e-01],
           [-1.25065278e+00],
           [-1.21139819e+00],
           [-1.08464500e+00],
           [ 3.76611232e-01],
           [ 6.90233825e-02],
           [-1.00621037e+00],
           [ 1.61180301e+00],
           [ 3.85839651e-01],
           [-2.73524960e-01],
           [-9.52663022e-01],
           [ 6.94553667e-01],
           [-8.11977651e-01],
           [-9.29365805e-01],
           [ 2.26268233e+00],
           [-5.19954223e-01],
           [ 1.33036550e+00],
           [ 9.27956516e-01],
           [-1.11019741e+00],
           [-1.79405710e-01],
           [ 2.40850957e+00],
           [ 4.94255961e-01],
           [ 6.82468165e-01],
           [ 2.51127913e-02],
           [ 7.17907816e-01],
           [ 1.08233784e+00],
           [-2.37246369e-01],
           [-5.24960347e-01],
           [-1.13077068e+00],
           [-1.37445699e+00],
           [-3.33677937e-01],
           [-1.06817669e+00],
           [-4.22789617e-01],
           [-6.94525764e-01],
           [-9.74295591e-01],
           [ 1.53459315e+00],
           [-1.21968033e+00],
           [ 4.70716415e-01],
           [-9.52288532e-01],
           [-6.03862727e-02],
           [ 2.04184480e+00],
           [-1.17072559e-01],
           [ 4.46897121e-01],
           [-3.65001105e-01],
           [-1.09637331e+00],
           [ 8.66255038e-01],
           [-3.57996545e-01],
           [-3.64558972e-01],
           [-1.40280473e+00],
           [-9.30191426e-01],
           [-1.11477016e+00],
           [ 7.22100440e-01],
           [-1.12496877e+00],
           [-1.05298892e+00],
           [-1.07624333e+00],
           [ 5.50475157e-01],
           [-1.31248264e+00],
           [-8.12584885e-02],
           [ 1.36645634e+00],
           [-8.28405184e-01],
           [-1.01573096e+00],
           [-1.20364915e+00],
           [ 4.26548677e-01],
           [ 1.42609094e-02],
           [-9.53057793e-01],
           [-3.76436243e-01],
           [-8.99919426e-01],
           [ 1.56845915e+00],
           [ 4.33541186e-01],
           [ 6.59208973e-01],
           [ 1.49284079e+00],
           [ 3.36700941e-01],
           [ 2.29926321e-01],
           [ 1.06670251e+00],
           [-1.36434466e+00],
           [ 2.07943431e+00],
           [ 1.07249733e+00],
           [-1.34578735e+00],
           [-8.61937875e-01],
           [-8.56100561e-01],
           [-7.13504516e-01],
           [ 8.50080330e-01],
           [-2.82518261e-01],
           [ 3.74011384e-01],
           [-5.73875008e-01],
           [-4.16293120e-01],
           [-1.09377608e+00],
           [-1.43160000e+00],
           [-4.21753677e-01],
           [ 4.26770065e-01],
           [ 4.27954629e+00],
           [-9.63856885e-01],
           [ 1.06656640e+00],
           [ 1.77199246e+00],
           [-4.40942391e-01],
           [-3.45513327e-01],
           [-5.66579097e-01],
           [ 1.49707217e+00],
           [-9.87353426e-01],
           [-4.59046178e-01],
           [-6.44369663e-01],
           [ 1.88956932e+00],
           [ 2.43681775e-01],
           [ 7.16011978e-01],
           [ 9.10175662e-01],
           [ 2.96313579e-01],
           [ 8.10777502e-02],
           [-7.86654540e-01],
           [-7.69202349e-01],
           [-4.46658648e-01],
           [-2.49548386e-01],
           [-6.04190603e-01],
           [-1.28130092e+00],
           [-6.99456372e-01],
           [ 1.54562672e+00],
           [-7.16873395e-02],
           [-1.20277164e+00],
           [ 1.53521693e+00],
           [ 1.39089381e+00],
           [-1.35980224e+00],
           [-8.96710739e-01],
           [-1.24866254e+00],
           [-6.65837078e-01],
           [ 1.88440309e+00],
           [ 5.33411131e-01],
           [ 1.71884678e+00],
           [ 9.95266673e-01],
           [-1.09652052e+00],
           [-2.15695619e-01],
           [ 3.56894301e-01],
           [ 8.37707886e-01],
           [ 5.80879746e-01],
           [ 5.16393698e-02],
           [ 4.83172855e-01],
           [-5.01813107e-01],
           [ 5.21163778e-01],
           [ 9.34277364e-01],
           [ 1.46131329e+00],
           [-1.77935330e-01],
           [ 2.32728312e-02],
           [ 5.83262313e-01],
           [-5.61722041e-01],
           [-3.58121155e-01],
           [ 7.84988177e-01],
           [ 3.24890848e-01],
           [-7.07236101e-01],
           [ 3.35927878e-01],
           [-2.95927396e-01],
           [-1.09861431e+00],
           [-9.48910800e-01],
           [-1.05988813e+00],
           [ 1.15987733e+00],
           [-5.09003064e-01],
           [-6.95420885e-01],
           [-8.65470424e-01],
           [-1.32492590e+00],
           [-1.44302733e+00],
           [ 6.57618322e-01],
           [-2.79653697e-01],
           [-1.12037595e+00],
           [-4.08811883e-01],
           [-1.33424353e+00],
           [-1.06310057e+00],
           [ 1.79696738e+00],
           [-1.39395891e-01],
           [-1.29787604e+00],
           [-3.18728102e-01],
           [ 4.52751520e-01],
           [-1.29600891e-01],
           [-8.32324738e-01],
           [-7.60228953e-01],
           [ 1.81740851e+00],
           [ 5.70591518e-01],
           [ 1.29876939e+00],
           [-9.75751777e-01],
           [ 6.96539053e-01],
           [-7.24138402e-01],
           [-8.13114297e-01],
           [-1.12917211e-01],
           [-2.26796453e-01],
           [-3.60462679e-01],
           [-8.73559875e-01],
           [ 7.45289669e-01],
           [-6.55398712e-01],
           [ 2.83733583e-01],
           [-6.78148128e-01],
           [-7.55276458e-01],
           [ 2.73580766e-01],
           [-9.53450921e-01],
           [-1.12709414e+00],
           [ 5.94870173e-01],
           [-1.09243112e+00],
           [ 1.55556925e+00],
           [-7.54234805e-01],
           [-8.48646424e-01],
           [-9.04407705e-01],
           [-1.69796178e-01],
           [ 1.20144097e+00],
           [-1.07250165e+00],
           [ 2.02747444e+00],
           [ 1.65962246e+00],
           [-9.32381133e-01],
           [-4.34755098e-01],
           [ 6.69963523e-01],
           [ 4.44147797e+00],
           [-1.34213587e+00],
           [ 1.85262192e+00],
           [ 9.52050470e-01],
           [-2.12424537e-01],
           [-5.10164436e-01],
           [ 7.73542700e-02],
           [-4.50357331e-01],
           [-1.27261371e+00],
           [-5.08833914e-02],
           [-9.35173145e-01],
           [ 6.66317951e-01],
           [-1.05948720e+00],
           [ 5.92865632e-01],
           [ 1.49403462e+00],
           [-2.89076278e-01],
           [ 1.14518341e-01],
           [-7.59965469e-01],
           [ 2.01359154e+00],
           [-3.24496918e-02],
           [ 5.83281397e-01],
           [ 1.01934377e+00],
           [-1.00877143e-01],
           [-8.99181818e-01],
           [-2.76971477e-01],
           [-9.38599257e-01],
           [ 5.26489389e-01],
           [ 7.14624238e-01],
           [-9.13908979e-01],
           [-8.00457624e-01],
           [ 4.36911421e-01],
           [-6.61842723e-01],
           [-2.09909683e-01],
           [ 1.23103149e+00],
           [-1.39711857e+00],
           [ 5.14519195e-01],
           [ 3.67989422e-01],
           [ 3.33611668e-01],
           [-1.00518273e-01],
           [-2.88146917e-01],
           [-4.88367982e-01],
           [ 1.95718898e+00],
           [ 1.57631909e-02],
           [-3.94106914e-02],
           [ 6.85126176e-01],
           [-1.30486012e+00],
           [ 8.01063571e-01],
           [-3.00282923e-01],
           [ 7.07098405e-01],
           [ 1.80615513e+00],
           [ 2.14183591e+00],
           [-6.08726263e-02],
           [ 1.32184457e-01],
           [-1.07819451e+00],
           [-5.97482627e-02],
           [ 5.36898245e-01],
           [ 8.64544740e-01],
           [ 2.02248565e-01],
           [-3.41781634e-01],
           [-9.50358472e-02],
           [-1.35710922e+00],
           [ 1.92852186e+00],
           [-4.30788521e-01],
           [ 7.74500181e-01],
           [-9.68931651e-01],
           [-8.13188956e-01],
           [-1.41081733e+00],
           [ 3.64070513e+00],
           [-6.12550788e-02],
           [-6.72375368e-01],
           [-1.35775068e+00],
           [-1.26002931e+00],
           [-6.34702843e-01],
           [-1.41906888e-01],
           [-7.40433239e-02],
           [-1.23569252e-01],
           [ 3.69180821e-01],
           [ 2.95993039e-01],
           [ 1.57606044e+00],
           [-6.95709237e-01],
           [ 1.07853977e+00],
           [-4.75812515e-01],
           [-1.50722505e-01],
           [-7.11628922e-01],
           [-1.07281398e+00],
           [-5.43660779e-01],
           [-2.73825666e-01],
           [-2.89632310e-01],
           [-1.58891114e-01],
           [-9.74556468e-01],
           [ 1.06147375e-01],
           [ 3.68924960e-01],
           [ 5.13023108e-01],
           [ 1.19513933e+00],
           [ 1.44626974e+00],
           [-5.84596942e-01],
           [-1.00830112e+00],
           [-9.72526683e-01],
           [-1.02369311e+00],
           [ 6.29747920e-01],
           [ 8.11777721e-01],
           [ 2.02934816e+00],
           [ 7.80715038e-01],
           [-7.93438818e-01],
           [-1.28096213e+00],
           [-9.49624718e-01],
           [ 5.17563819e-01],
           [-1.05881304e+00],
           [ 1.67066276e+00],
           [-3.52940846e-01],
           [-5.85845023e-01],
           [ 4.85328392e-01],
           [-2.12354644e-01],
           [ 1.06445089e+00],
           [ 6.90487652e-01],
           [-8.09880830e-01],
           [ 1.19707080e+00],
           [ 1.59393636e+00],
           [-1.33612671e+00],
           [ 6.93022340e-01],
           [-1.05672670e+00],
           [ 5.68289698e-01],
           [-7.83896930e-01],
           [ 5.92225158e-01],
           [-1.18818738e+00],
           [ 3.44302057e-01],
           [-8.63800187e-01],
           [-2.79740513e-01],
           [ 6.88518209e-01],
           [ 7.69019968e-01],
           [-2.01574797e-01],
           [ 6.61400251e-01],
           [ 1.29379547e+00],
           [ 5.87883877e-01],
           [-4.64862069e-01],
           [ 7.86731876e-01],
           [-3.31958266e-01],
           [-7.66033704e-01],
           [-1.07060253e+00],
           [ 1.15686284e+00],
           [-1.15539039e+00],
           [-6.90025239e-01],
           [-1.91231066e-01],
           [-8.19344436e-01],
           [-1.29150090e+00],
           [-9.23526295e-01],
           [-2.64823509e-01],
           [ 3.41309045e-01],
           [ 1.49138825e+00],
           [ 1.14931391e+00],
           [-1.96895873e-01],
           [ 1.97343740e+00],
           [-9.58236569e-02],
           [ 1.16176986e+00],
           [-7.77796417e-01],
           [ 1.85659573e+00],
           [-2.23533868e-01],
           [ 1.50771724e-01],
           [-1.41639810e+00],
           [-8.48666669e-01],
           [-1.25789191e+00],
           [ 3.85766795e-01],
           [ 4.70868519e+00],
           [-5.72655152e-01],
           [ 6.56049397e-01],
           [-1.19002206e+00],
           [-8.73566016e-01],
           [-6.63570856e-01],
           [ 1.98160912e+00],
           [ 2.10834480e+00],
           [-6.91436188e-01],
           [-7.13866724e-01],
           [ 6.36459182e-01],
           [ 1.53008350e+00],
           [-3.60054376e-01],
           [-9.46728859e-01],
           [ 6.92306814e-01],
           [ 1.06366083e+00],
           [-7.61313916e-01],
           [-2.27828287e-01],
           [-9.21674855e-01],
           [ 1.77114813e+00],
           [-4.25091884e-01],
           [ 7.90703220e-01],
           [-9.50739138e-01],
           [-1.02265178e+00],
           [ 9.13253777e-01],
           [ 8.39947151e-01],
           [-2.15842776e-01],
           [-1.73554720e-01],
           [-4.84639771e-01],
           [-3.75317789e-01],
           [-3.88878117e-02],
           [ 8.53705676e-01],
           [ 9.92792113e-01],
           [-1.02624811e+00],
           [ 6.46979276e-01],
           [-1.25517982e+00],
           [ 5.83506230e-01],
           [-4.90610175e-01],
           [-1.27259638e+00],
           [-5.06526772e-01],
           [ 7.13302676e-01],
           [-1.09600216e+00],
           [-1.11835196e+00],
           [ 1.60784376e+00],
           [ 1.63882569e+00],
           [-9.78038994e-01],
           [ 6.28736669e-01],
           [ 6.76084548e-01],
           [-1.61055613e-01],
           [ 1.62529852e+00],
           [-9.12057179e-03],
           [ 3.91830710e-01],
           [-4.65728965e-01],
           [-8.77025852e-01],
           [ 1.15071982e+00],
           [-6.94370198e-01],
           [-1.06042885e-01],
           [ 1.42650945e+00],
           [-1.19652014e+00],
           [-1.27582992e+00],
           [-1.06211013e+00],
           [-8.63252831e-01],
           [ 1.77733814e+00],
           [ 2.78799689e+00],
           [ 6.88027213e-01],
           [ 2.01591942e+00],
           [ 2.90429045e-01],
           [-1.29529681e+00],
           [ 2.77821949e-01],
           [ 9.82097493e-01],
           [ 2.56831229e-01],
           [ 3.50430527e-01],
           [ 2.28094572e+00],
           [-1.19860215e+00],
           [-6.68537900e-01],
           [ 1.28398640e+00],
           [-8.22396130e-01],
           [-1.18637502e-01],
           [-1.33124957e+00],
           [-1.49127855e-01],
           [-1.30562215e+00],
           [-9.38066719e-01],
           [ 6.61766814e-01],
           [ 3.46258219e-01],
           [ 1.85591860e+00],
           [-2.07603043e-01],
           [ 4.29877262e-01],
           [-6.41520576e-01],
           [ 6.91919078e-01],
           [ 6.38213147e-01],
           [ 6.71923327e-01],
           [-1.31007947e+00],
           [-1.17549561e+00],
           [ 2.80343475e+00],
           [ 1.71274216e-01],
           [ 1.44944817e-01],
           [-1.03712645e+00],
           [ 1.25987420e+00],
           [-5.69384534e-01],
           [-4.02780888e-01],
           [ 1.29595738e+00],
           [ 1.39434868e+00],
           [-4.80776578e-01],
           [ 2.27624384e+00],
           [-1.10477602e+00],
           [-8.16408160e-02],
           [ 9.31753743e-02],
           [ 8.14633359e-01],
           [ 4.34691686e-01],
           [-1.09887738e+00],
           [-5.97547698e-01],
           [-1.01768741e+00],
           [-7.63803151e-01],
           [-2.26763908e-01],
           [ 1.03820038e+00],
           [ 1.89489962e+00],
           [-2.95181987e-02],
           [-2.18482168e-01],
           [-8.68986228e-02],
           [ 7.37148392e-01],
           [-1.09285661e+00],
           [ 3.86659362e-01],
           [ 3.91133966e-01],
           [ 4.49353739e-01],
           [-8.89412400e-01],
           [-8.99298858e-01],
           [-1.20964455e+00],
           [ 1.70242255e+00],
           [-3.32725349e-01],
           [ 2.33769431e-01],
           [ 1.61964785e+00],
           [ 6.58878007e-01],
           [ 1.63550280e+00],
           [ 1.85968830e+00],
           [-7.95682993e-01],
           [ 1.66079871e+00],
           [-1.07271376e+00],
           [ 3.91924882e-01],
           [-8.99922765e-01],
           [-9.14378230e-01],
           [ 1.97135791e+00],
           [-1.22244374e+00],
           [ 6.17769121e-01],
           [ 7.60272441e-01],
           [ 9.69993460e-01],
           [-9.15761561e-01],
           [-2.67713353e-01],
           [ 2.84346010e-01],
           [-8.60314019e-01],
           [-9.49968592e-01],
           [-6.77890196e-01],
           [-1.30391589e+00],
           [-2.21356694e-01],
           [-6.50302987e-01],
           [ 3.39517322e-01],
           [ 1.26331288e+00],
           [-1.02861743e-01],
           [ 1.52660885e+00],
           [ 4.86166778e-01],
           [ 1.07694086e+00],
           [-2.06166637e-01],
           [ 4.88266330e-01],
           [-1.10383816e+00],
           [-5.55309611e-01],
           [ 6.10567168e-01],
           [-1.68044284e-01],
           [ 8.70313017e-02],
           [-1.31871562e+00],
           [-1.21661041e+00],
           [-1.04057532e+00],
           [ 9.28576227e-01],
           [ 2.26801076e-01],
           [ 4.05965385e-01],
           [-1.87985996e-01],
           [-4.24903023e-01],
           [-7.38542773e-01],
           [-5.75632794e-01],
           [ 1.60877492e+00],
           [-9.38539755e-01],
           [ 2.65907972e-01],
           [ 6.31467037e-01],
           [-1.28564850e+00],
           [ 5.71867074e-01],
           [-3.04628989e-01],
           [ 6.76690888e-01],
           [ 7.61154797e-01],
           [-1.37619355e+00],
           [ 1.60591716e+00],
           [ 1.88050423e+00],
           [-1.12061708e+00],
           [ 1.04137113e+00],
           [ 6.29568503e-01],
           [ 1.03444782e+00],
           [ 3.58110390e-01],
           [-1.81969408e-02],
           [ 1.69873434e+00],
           [-8.61883372e-01],
           [-7.37767515e-02],
           [-1.00663354e+00],
           [-1.07034546e+00],
           [-5.76279587e-01],
           [-1.18290658e+00],
           [-1.02006153e+00],
           [-7.53597937e-01],
           [-4.41549588e-01],
           [-7.68606489e-01],
           [-9.10551331e-01],
           [ 1.61943110e-01],
           [ 1.63869333e+00],
           [-1.01350534e+00],
           [-1.31640821e+00],
           [ 2.38478449e+00],
           [-4.79516750e-01],
           [-7.21583773e-01],
           [-1.19593182e+00],
           [ 4.02194846e-01],
           [ 5.92388151e-01],
           [ 1.63936230e+00],
           [-2.13383890e-01],
           [-6.47218071e-01],
           [-8.71984881e-01],
           [-1.14323932e+00],
           [-1.28732015e+00],
           [-4.05201212e-01],
           [ 4.90765562e-01],
           [ 9.56057322e-01],
           [-5.77642102e-01],
           [-1.12366153e+00],
           [ 7.46532751e-01],
           [-6.19323462e-02],
           [ 3.02678706e+00],
           [-2.07956735e-01],
           [-5.45911167e-01],
           [ 1.01180205e-01],
           [-2.47289322e-01],
           [-1.08445500e-01],
           [-2.46687356e-01],
           [-1.41905272e+00],
           [ 3.55969847e-02],
           [-2.61997810e-01],
           [ 9.01019832e-01],
           [-9.47104259e-01],
           [-6.28995976e-01],
           [-3.01537824e-01],
           [ 8.98592795e-01],
           [ 5.08479362e-02],
           [ 7.75792071e-01],
           [-7.06475374e-01],
           [ 2.33872136e-01],
           [-5.65147350e-01],
           [-1.10085498e+00],
           [-1.33743371e+00],
           [-1.17871335e+00],
           [ 5.23848408e-01],
           [-1.36204206e+00],
           [-7.02800256e-01],
           [ 1.27832479e+00],
           [-6.69491255e-01],
           [-1.29724768e+00],
           [-1.08193953e+00],
           [-9.57554658e-01],
           [-1.15447358e+00],
           [-4.67057043e-01],
           [-9.56729626e-01],
           [ 1.26166104e+00],
           [ 4.80933375e-01],
           [ 3.85182967e-01],
           [ 1.88657864e+00],
           [ 7.25559526e-01],
           [ 5.27973853e-01],
           [ 9.12947376e-01],
           [ 1.72801935e+00],
           [ 4.84262424e-01],
           [-3.17532972e-01],
           [-1.03001399e+00],
           [-1.43629558e-01],
           [-2.40234222e-01],
           [-5.81236188e-01],
           [ 5.13127474e-01],
           [ 6.20629223e-01],
           [ 1.63503366e+00],
           [ 5.07749699e-01],
           [ 1.43238077e+00],
           [-7.19391692e-01],
           [-1.06128515e+00],
           [-1.10565496e+00],
           [ 1.27495143e-01],
           [ 2.59520768e-01],
           [-4.52224213e-01],
           [-6.35384537e-01],
           [ 7.21221726e-01],
           [ 5.79003045e-01],
           [ 1.48727782e+00],
           [ 2.06756063e+00],
           [-6.39684758e-01],
           [-1.27464282e+00],
           [ 1.45968470e+00],
           [ 2.19789676e+00],
           [ 1.68027217e-01],
           [-1.27101808e+00],
           [-5.93810043e-01],
           [-2.55031437e-01],
           [-9.51373845e-01],
           [ 3.69542886e-01],
           [-7.56589718e-01],
           [-5.16347694e-01],
           [-3.46939575e-01],
           [ 1.50356289e+00],
           [ 4.06574742e-01],
           [-2.56487445e-01],
           [ 2.89032682e-01],
           [-6.13109710e-01],
           [-1.07962115e+00],
           [-1.10318826e-01],
           [-8.69397118e-01],
           [-9.69166554e-01],
           [-1.31189272e+00],
           [ 5.88208969e-01],
           [-6.13115691e-01],
           [-1.18026182e-01],
           [-3.00657824e-01],
           [-1.08966808e-01],
           [-7.11926844e-01],
           [ 1.94012206e-01],
           [-7.31595252e-01],
           [-9.56864253e-02],
           [ 1.15271610e+00],
           [-4.56053308e-01],
           [-5.66071892e-01],
           [-9.14585443e-01],
           [-6.61782257e-01],
           [ 2.47936579e+00],
           [-7.24072634e-01],
           [-1.14587496e+00],
           [-7.68435177e-01],
           [-1.08336191e+00],
           [ 7.87631107e-03],
           [-7.74348471e-01],
           [ 1.51311004e+00],
           [-5.77372585e-02],
           [-2.94895687e-01],
           [-1.32949916e-01],
           [-9.05147420e-01],
           [ 1.49064989e+00],
           [-4.07194169e-02],
           [-7.87621035e-01],
           [ 8.88553698e-01],
           [-8.36682604e-01],
           [-7.72151712e-01],
           [ 1.00316273e-01],
           [-3.08210418e-01],
           [-7.15074762e-01],
           [ 3.31424461e-01],
           [-3.86075124e-01],
           [ 1.96086392e+00],
           [ 1.35287323e+00],
           [-1.30272159e+00],
           [-1.45322558e+00],
           [-3.51472216e-01],
           [ 1.66796542e+00],
           [-1.55085905e-01],
           [-2.88804118e-01],
           [-8.44672476e-02],
           [-1.41989498e+00],
           [ 7.29223111e-01],
           [-9.08644978e-01],
           [-9.59674205e-01],
           [-1.97383494e-01],
           [-8.73519511e-01],
           [-6.55434470e-01],
           [ 1.82207180e+00],
           [ 7.96516291e-01],
           [-1.11563572e+00],
           [-4.84821580e-01],
           [ 6.95155383e-01],
           [-1.35274502e+00],
           [ 4.08446854e-01],
           [ 1.53854022e+00],
           [ 4.87124364e-01],
           [-1.28557327e-01],
           [-1.06210631e+00],
           [ 7.60381000e-02],
           [-5.61252380e-01],
           [-7.97089978e-01],
           [-1.24401745e+00],
           [-4.56180007e-01],
           [-1.19013947e+00],
           [-1.78923766e-01],
           [ 3.11492818e-01],
           [ 4.37421911e-01],
           [-1.32285020e-01],
           [-8.34846393e-01],
           [-1.02255607e+00],
           [-8.40699253e-03],
           [ 1.34226350e+00],
           [ 2.09330349e+00],
           [ 3.29592176e-02],
           [ 1.17062415e+00]])




```python
#In: 
from sklearn.linear_model import LinearRegression # sem regularizar
from sklearn.linear_model import Lasso # com regularização l1
from sklearn.linear_model import Ridge # com regularização l2
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import GridSearchCV
```

O gridsearch cv faz o laço que fizemos na mão acima (conjuntos de validação)


```python
#In: 
linear = LinearRegression(fit_intercept=False)
lasso = GridSearchCV(Lasso(fit_intercept=False),
                     cv=10,
                     refit=True,
                     param_grid={'alpha': [0.001, 0.01, 0.1, 1, 10, 100]})

ridge = GridSearchCV(Ridge(fit_intercept=False),
                     cv=10,
                     refit=True,
                     param_grid={'alpha': [0.001, 0.01, 0.1, 1, 10, 100]})

knn = GridSearchCV(KNeighborsRegressor(),
                   cv=10,
                   refit=True,
                   param_grid={'n_neighbors': [2, 3, 5, 7, 11, 13],
                               'weights': ['uniform', 'distance']})
```


```python
#In: 
linear = linear.fit(X_train, y_train)
linear.score(X_train, y_train)
```




    0.9316391152834318




```python
#In: 
lasso = lasso.fit(X_train, y_train)
lasso.score(X_train, y_train)
```




    0.9318514675371551




```python
#In: 
ridge = ridge.fit(X_train, y_train)
ridge.score(X_train, y_train)
```




    0.9319223512617757




```python
#In: 
knn = knn.fit(X_train, y_train)
knn.score(X_train, y_train)
```




    1.0




```python
#In: 
knn.best_params_
```




    {'n_neighbors': 13, 'weights': 'distance'}



Agora vamos usar o Bootstrap para entender o erro dos modelos caso eu repita o experimento.


```python
#In: 
from sklearn.metrics import mean_squared_error
```


```python
#In: 
y_pred = linear.predict(X_train)
mean_squared_error(y_train, y_pred)
```




    0.06836088471656822




```python
#In: 
def bootstrap_score(X, y, model, n=1000):
    size = len(y)    
    samples = np.zeros(size)
    for i in range(size):
        # Gera amostras com reposição
        idx = np.random.choice(size, size)
        Xb = X[idx]
        yb = y[idx]
        
        err = mean_squared_error(yb, model.predict(Xb))
        samples[i] = err
    return samples
```


```python
#In: 
y_test_df = test_df['Weekly_Sales_Store']
X_test_df = test_df.drop('Weekly_Sales_Store', axis='columns')
```


```python
#In: 
X_test = scaler_x.transform(X_test_df.values)
y_test = scaler_y.transform(y_test_df.values[:, np.newaxis])
```


```python
#In: 
samples = bootstrap_score(X_test, y_test, knn)
plt.hist(samples, edgecolor='k')
plt.title('({}, {})'.format(ss.scoreatpercentile(samples, 2.5),
                            ss.scoreatpercentile(samples, 97.5)))
despine()
```


    
![png](21-pratica_files/21-pratica_90_0.png)
    



```python
#In: 
samples = bootstrap_score(X_test, y_test, lasso)
plt.hist(samples, edgecolor='k')
plt.title('({}, {})'.format(ss.scoreatpercentile(samples, 2.5),
                            ss.scoreatpercentile(samples, 97.5)))
despine()
```


    
![png](21-pratica_files/21-pratica_91_0.png)
    

