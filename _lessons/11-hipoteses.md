---
layout: page
title: Testes de Hipóteses
nav_order: 11
---

[<img src="./colab_favicon_small.png" style="float: right;">](https://colab.research.google.com/github/icd-ufmg/icd-ufmg.github.io/blob/master/_lessons/11-hipoteses.ipynb)

# Testes de Hipóteses
{: .no_toc .mb-2 }

Entendendo valores-p
{: .fs-6 .fw-300 }

{: .no_toc .text-delta }
Resultados Esperados


1. Entender o conceito de um teste de hipótese
1. Entender o valor-p
1. Saber realizar e interpretar testes de hipóteses
1. Saber realizar e interpretar valores-p

---
**Sumário**
1. TOC
{:toc}
---


```python
#In: 
# -*- coding: utf8

from scipy import stats as ss

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```


```python
#In: 
plt.style.use('seaborn-colorblind')

plt.rcParams['figure.figsize']  = (16, 10)
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

## Introdução

Este é o primeiro notbeook com base em testes de hipóteses. Para entender tal mundo, temos que cobrir:

1. Testes quando sabemos da população
1. Testes quando não sabemos da população
1. Erros e Poder de Teste (próximo notebook)
1. Causalidade (próximo notebook)

Assim como no mundo de intervalos de confiânça, a nossa ideia aqui é ter algum garantia estatística para fazer afirmações quando temos apenas uma amostra. Note que tal amostra, idealmente, vai ser grande o suficiente para estimar com menor viés alguns parâmetros como a média e o desvio padrão da população.

Quando testamos hipóteses queremos responder perguntas como: "Tal cenário é factível quando comparamos com uma __hipótese nula?__". Esta __hipótese nula__ representa um modelo probabílistico da população. No geral, queremos rejeitar a mesma indicando que não temos evidências de que os dados seguem tal modelo. Modelos, para fins deste notebook, quer dizer formas de amostrar a população. Imagine que uma moeda é jogada para cima 100 vezes e cai em caras 99 vezes. Uma __hipótese nula__ seria: "A moeda é justa!". Qual a chance de uma moeda justa cair em caras 99 vezes? 


```python
#In: 
x = np.arange(0, 101) # Valores no eixo x
prob_binom = ss.distributions.binom.pmf(x, 100, 0.5)
plt.plot(x, prob_binom, 'o')
plt.plot([99], prob_binom[99], 'ro')
plt.text(99, 0.0018, 'Aqui', horizontalalignment='center')
plt.xlabel('Num Caras - x')
plt.ylabel('P(sair x caras)')
plt.title('Chance de sair 99 caras (pontinho vermelho!)')
despine()
```


    
![png](11-hipoteses_files/11-hipoteses_6_0.png)
    


Quase zero! Porém, é fácil falar de moedas. Foi assim também quando estudamos o IC. Vamos explorar outros casos.

## Exemplo 1: Juri no Alabama

__No início dos anos 1960, no Condado de Talladega, no Alabama, um negro chamado Robert Swain foi condenado por  à morte por estuprar uma mulher branca. Ele recorreu da sentença, citando entre outros fatores o júri todo branco. Na época, apenas homens com 21 anos ou mais eram autorizados a servir em júris no condado de Talladega. No condado, 26% dos jurados elegíveis eram negros. No juri final, havia apenas 8 negros entre os 100 selecionados para o painel de jurados no julgamento de Swain.__

Nossa pergunta: **Qual é a probabilidade (chance) de serem selecionados 8 indíviduos negros?**

Para nos ajudar com este exemplo, o código amostra proporções de uma população. O mesmo gera 10,000 amostras ($n$) de uma população de tamanho `pop_size`. Tais amostras são geradas sem reposição. Além do mais, o código assume que:

1. [0, $pop\_size * prop$) pertencem a um grupo.
1. [$pop\_size * prop$, $pop\_size$) pertencem a outro grupo.

Ou seja, em uma população de 10 (pop_size) pessoas. Caso a proporção seja 0.2 (prop). A população tem a seguinte cara:

__[G1, G1, G2, G2, G2, G2, G2, G2, G2, G2]__.

A ideia do cógido é responder: **Ao realizar amostras uniformes da população acima, quantas pessoas do tipo G1 e do tipo G2 são amostradas**. Para isto, realizamos 10,0000 amostras.


```python
#In: 
def sample_proportion(pop_size, prop, n=10000):
    '''
    Amostra proporções de uma população.
    
    Parâmetros
    ----------
    pop_size: int, tamanho da população
    prop: double, entre 0 e 1
    n: int, número de amostras
    '''
    assert(prop >= 0)
    assert(prop <= 1)
    
    grupo = pop_size * prop
    resultados = np.zeros(n)
    for i in range(n):
        sample = np.random.randint(0, pop_size, 100)
        resultados[i] = np.sum(sample < grupo)
    return resultados
```

Vamos ver agora qual é a cara de 10,000 amostras da cidade de Talladega. Vamos assumir que a cidade tem uma população de 100,000 habitantes. Tal número não importa muito para o exemplo, estamos olhando amostras. Podemos ajustar com a real.

O gráfico abaixo mostra no eixo-x o número de pessoas negras em cada amostra uniforme. Realizamos 10,000 delas. No eixo-y, a quantidade de amostras com aquele percentua. Agora responda, qual a chance de sair 8 pessoas apenas?


```python
#In: 
proporcoes = sample_proportion(pop_size=100000, prop=0.26)
bins = np.linspace(1, 100, 100) + 0.5
plt.hist(proporcoes, bins=bins, edgecolor='k')
plt.xlim(0, 52)
plt.ylabel('Numero de Amostras de Tamanho 10k')
plt.xlabel('Número no Grupo')
plt.plot([8], [0], 'ro', ms=15)
despine()
```


    
![png](11-hipoteses_files/11-hipoteses_11_0.png)
    


Podemos usar 5\% de chance para nos ajudar. É assim que funciona os testes. Com 5\% de chances, saiem pelo menos 19 pessoas negras. Então estamos abaixo disto, bem raro!


```python
#In: 
np.percentile(proporcoes, 5)
```




    19.0



Este exemplo, assim como o próximo, embora não assuma nada sobre a população, assume uma hipótese nula de seleção uniforme. Isto é, qual é a probabilidade de ocorre o valor que observamos em uma seleção uniforme?!

## Ideia de Testes de Hipóteses

1. Dado um valor observado $t_{obs}.$
1. Qual é a chance deste valor em uma hipótese (modelo) nulo?!

No exemplo acima $t_{obs}=8$.

## Exemplo 2 -- Um outro Juri

In 2010, the American Civil Liberties Union (ACLU) of Northern California presented a report on jury selection in Alameda County, California. The report concluded that certain ethnic groups are underrepresented among jury panelists in Alameda County, and suggested some reforms of the process by which eligible jurors are assigned to panels. In this section, we will perform our own analysis of the data and examine some questions that arise as a result.

Aqui temos um outro exemplo de juri. Neste caso, temos diferentes grupos raciais. Como podemos falar algo dos grupos? Precisamos de uma estatística de teste. Portanto, vamos usar a __total variation distance__.

$$TVD(p, q) = \sum_{i=0}^{n-1} abs(p_i - q_i)$$

p e q aqui são vetores com proporções. Cada vetor deve somar 1. Quão maior a TVD, maior a diferença entre as proporções dos dois vetores! Quando p == q, temos TVD == 0.


```python
#In: 
def total_variation(p, q):
    '''
    Computa a total variation distance com base em dois vetore, p e q
    
    Parâmetros
    ----------
    p: vetor de probabilidades de tamanho n
    q: vetor de probabilidades de tamanho n
    '''
    return np.sum(np.abs(p - q)) / 2
```

Nossos dados. No juri, temos as seguintes proporções em cada raça.


```python
#In: 
idx = ['Asian', 'Black', 'Latino', 'White', 'Other']
df = pd.DataFrame(index=idx)
df['pop'] = [0.15, 0.18, 0.12, 0.54, 0.01]
df['sample'] = [0.26, 0.08, 0.08, 0.54, 0.04]
df
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
      <th>pop</th>
      <th>sample</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Asian</th>
      <td>0.15</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>Black</th>
      <td>0.18</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>Latino</th>
      <td>0.12</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>White</th>
      <td>0.54</td>
      <td>0.54</td>
    </tr>
    <tr>
      <th>Other</th>
      <td>0.01</td>
      <td>0.04</td>
    </tr>
  </tbody>
</table>
</div>




```python
#In: 
df.plot.bar()
plt.ylabel('Propopção')
plt.ylabel('Grupo')
despine()
```


    
![png](11-hipoteses_files/11-hipoteses_20_0.png)
    


Vamos comparar com uma amostra aleatória! Para isto, vou amostrar cada grupo por vez. Note a diferença.


```python
#In: 
N = 1453
uma_amostra = []
for g in df.index:
    p = df.loc[g]['pop']
    s = sample_proportion(N, p, 1)[0]
    uma_amostra.append(s/100)
```


```python
#In: 
df['1random'] = uma_amostra
df.plot.bar()
plt.ylabel('Propopção')
plt.ylabel('Grupo')
despine()
```


    
![png](11-hipoteses_files/11-hipoteses_23_0.png)
    


Agora compare o TVD nos dados e na amostra aleatória!


```python
#In: 
total_variation(df['1random'], df['pop'])
```




    0.029999999999999995




```python
#In: 
total_variation(df['sample'], df['pop'])
```




    0.14



Para realizar o teste, fazemos 10,000 amostras e comparamos os TVDs! O código abaixo guarda o resultado de cada amostra em uma linha de uma matriz.


```python
#In: 
N = 1453
A = np.zeros(shape=(10000, len(df.index)))
for i, g in enumerate(df.index):
    p = df.loc[g]['pop']
    A[:, i] = sample_proportion(N, p) / 100
```


```python
#In: 
A
```




    array([[0.18, 0.25, 0.15, 0.51, 0.01],
           [0.12, 0.16, 0.17, 0.52, 0.05],
           [0.16, 0.27, 0.14, 0.54, 0.01],
           ...,
           [0.13, 0.18, 0.09, 0.54, 0.02],
           [0.18, 0.15, 0.14, 0.48, 0.  ],
           [0.13, 0.21, 0.19, 0.53, 0.01]])



Agora o histograma da TVD. O  ponto vermelho mostra o valor que observamos, as barras mostram as diferentes amostras. Novamente, bastante raro tal valor. Rejeitamos a hipótese nula e indicamos que os dados não foram selecionados de forma uniforme!


```python
#In: 
all_distances = []
for i in range(A.shape[0]):
    all_distances.append(total_variation(df['pop'], A[i]))
```


```python
#In: 
plt.hist(all_distances, bins=30, edgecolor='k')
plt.ylabel('Numero de Amostras de Tamanho 10k')
plt.xlabel('Total Variation Distance')
plt.plot([0.14], [0], 'ro', ms=15)
despine()
```


    
![png](11-hipoteses_files/11-hipoteses_32_0.png)
    



```python
#In: 
np.percentile(all_distances, 97.5)
```




    0.12000000000000001



## Caso 3. Dados Reais

Agora, finalmente vamos assumir um caso com dados reais. Em particular, vamos comparar salários de dois times da NBA. O dataframe se encontra abaixo. Vamos focar nos times:

1. Houston Rockets
1. Cleveland Cavaliers

Diferente do exemplo anterior, simular aqui vair ser um pouco mais complicado. É mais complicado assumir uma população para gerar amostras. Portanto vamos fazer uso de __testes de permutação__. Inicialmente, vamos explorar os dados (abaixo)


```python
#In: 
df = pd.read_csv('https://media.githubusercontent.com/media/icd-ufmg/material/master/aulas/11-Hipoteses/nba_salaries.csv')
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
      <th>PLAYER</th>
      <th>POSITION</th>
      <th>TEAM</th>
      <th>SALARY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Paul Millsap</td>
      <td>PF</td>
      <td>Atlanta Hawks</td>
      <td>18.671659</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Al Horford</td>
      <td>C</td>
      <td>Atlanta Hawks</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tiago Splitter</td>
      <td>C</td>
      <td>Atlanta Hawks</td>
      <td>9.756250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jeff Teague</td>
      <td>PG</td>
      <td>Atlanta Hawks</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kyle Korver</td>
      <td>SG</td>
      <td>Atlanta Hawks</td>
      <td>5.746479</td>
    </tr>
  </tbody>
</table>
</div>




```python
#In: 
df = df[df['TEAM'].isin(['Houston Rockets', 'Cleveland Cavaliers'])]
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
      <th>PLAYER</th>
      <th>POSITION</th>
      <th>TEAM</th>
      <th>SALARY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>72</th>
      <td>LeBron James</td>
      <td>SF</td>
      <td>Cleveland Cavaliers</td>
      <td>22.970500</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Kevin Love</td>
      <td>PF</td>
      <td>Cleveland Cavaliers</td>
      <td>19.689000</td>
    </tr>
    <tr>
      <th>74</th>
      <td>Kyrie Irving</td>
      <td>PG</td>
      <td>Cleveland Cavaliers</td>
      <td>16.407501</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Tristan Thompson</td>
      <td>C</td>
      <td>Cleveland Cavaliers</td>
      <td>14.260870</td>
    </tr>
    <tr>
      <th>76</th>
      <td>Brendan Haywood</td>
      <td>C</td>
      <td>Cleveland Cavaliers</td>
      <td>10.522500</td>
    </tr>
  </tbody>
</table>
</div>



Vamos pegar o salário médio de cada time. Observe que teremos uma diferença de mais ou menos 3 milhões e doláres. Tal estatística será a nossa $t_{obs}$. Antes disso, vamos criar um filtro (vetor de booleanos) para o Houston.


```python
#In: 
filtro = df['TEAM'] == 'Houston Rockets'
```

Agora o salário do Houston


```python
#In: 
filtro = df['TEAM'] == 'Houston Rockets'
df[filtro]['SALARY'].mean()
```




    7.107153083333334



Do não Houston, ou seja, Cleveland.


```python
#In: 
df[~filtro]['SALARY'].mean()
```




    10.231241200000001



Por fim, nossa estatística observada.


```python
#In: 
t_obs = df[~filtro]['SALARY'].mean() - df[filtro]['SALARY'].mean()
t_obs
```




    3.124088116666667



## Teste de Permutação

Embora não estamos comparando proporções como nos exemplos anteriores, podemos sim brincar um pouco de simulação. Vamos assumir um seguinte modelo nulo. __A probabilidade de um jogador escolher (ou for contratado) em um time para jogar é igual para os dois times__. Ou seja, vamos dizer que a associação Jogador x Time é uniformimente aleatória! Como fazemos isto? Basta fazer um shuffle no filtro acima.

Assumindo apenas 5 jogadores. Os dados reais são:

1. Nomes:    [J1, J2, J3, J4, J5]
1. Salários: [S1, S2, S3, S4, S5]
1. Times:    [T1, T1, T2, T2, T2]

Vamos manter os nomes e os salários fixos. Ao fazer um shuffle (embaralhamento) nos times, temos:

1. Nomes:    [J1, J2, J3, J4, J5]
1. Salários: [S1, S2, S3, S4, S5]
1. Times:    [T1, T2, T2, T2, T1]

Esta é nossa hipótese nula! Vamos realizar uma!


```python
#In: 
np.random.shuffle(filtro.values)
diff = df[~filtro]['SALARY'].mean() - df[filtro]['SALARY'].mean()
diff
```




    -2.9790110833333374



Note acima que temos uma diferença de salários diferente do que observamos. Vamos agora gerar 10,000!


```python
#In: 
N = 10000
diferencas = np.zeros(N)
for i in range(N):
    np.random.shuffle(filtro.values)
    diff = df[~filtro]['SALARY'].mean() - df[filtro]['SALARY'].mean()
    diferencas[i] = diff
```

Na figura abaixo mostramos os resultados do mundo simulado. Note que em 16% dos casos temos diferença de salários mais extremas do que $t_{obs}$. Isto é uma chance bem alta! Nos exemplos anteriores tinhamos valores bem menores. Ou seja, **não rejeitamos a hipótese nula (modelo simulado)**. A variação dos salários pode ser explicado pelo acaso.


```python
#In: 
plt.hist(diferencas, bins=50, edgecolor='k')
plt.xlabel('Diferença na Permutação')
plt.ylabel('Pr(diff)')
plt.vlines(t_obs, 0, 0.14, color='red')
plt.text(t_obs+1, 0.10, '$16\%$ dos casos')
despine()
plt.show()
```


    
![png](11-hipoteses_files/11-hipoteses_50_0.png)
    


## Animação


```python
#In: 
from IPython.display import HTML
from matplotlib import animation
```


```python
#In: 
def update_hist(num, data):
    plt.cla()
    plt.hist(data[0:100 * (num+1)], bins=50,
             density=True, edgecolor='k')
    plt.xlabel('Diferença na Permutação')
    plt.ylabel('Pr(diff)')
    despine()
```


```python
#In: 
fig = plt.figure()
ani = animation.FuncAnimation(fig, update_hist, 30, fargs=(diferencas, ))
HTML(ani.to_html5_video())
```




<video width="1152" height="720" controls autoplay loop>
  <source type="video/mp4" src="data:video/mp4;base64,AAAAIGZ0eXBNNFYgAAACAE00ViBpc29taXNvMmF2YzEAAAAIZnJlZQAA3dptZGF0AAACrQYF//+p
3EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2MCByMzAxMSBjZGU5YTkzIC0gSC4yNjQvTVBF
Ry00IEFWQyBjb2RlYyAtIENvcHlsZWZ0IDIwMDMtMjAyMCAtIGh0dHA6Ly93d3cudmlkZW9sYW4u
b3JnL3gyNjQuaHRtbCAtIG9wdGlvbnM6IGNhYmFjPTEgcmVmPTMgZGVibG9jaz0xOjA6MCBhbmFs
eXNlPTB4MzoweDExMyBtZT1oZXggc3VibWU9NyBwc3k9MSBwc3lfcmQ9MS4wMDowLjAwIG1peGVk
X3JlZj0xIG1lX3JhbmdlPTE2IGNocm9tYV9tZT0xIHRyZWxsaXM9MSA4eDhkY3Q9MSBjcW09MCBk
ZWFkem9uZT0yMSwxMSBmYXN0X3Bza2lwPTEgY2hyb21hX3FwX29mZnNldD0tMiB0aHJlYWRzPTYg
bG9va2FoZWFkX3RocmVhZHM9MSBzbGljZWRfdGhyZWFkcz0wIG5yPTAgZGVjaW1hdGU9MSBpbnRl
cmxhY2VkPTAgYmx1cmF5X2NvbXBhdD0wIGNvbnN0cmFpbmVkX2ludHJhPTAgYmZyYW1lcz0zIGJf
cHlyYW1pZD0yIGJfYWRhcHQ9MSBiX2JpYXM9MCBkaXJlY3Q9MSB3ZWlnaHRiPTEgb3Blbl9nb3A9
MCB3ZWlnaHRwPTIga2V5aW50PTI1MCBrZXlpbnRfbWluPTUgc2NlbmVjdXQ9NDAgaW50cmFfcmVm
cmVzaD0wIHJjX2xvb2thaGVhZD00MCByYz1jcmYgbWJ0cmVlPTEgY3JmPTIzLjAgcWNvbXA9MC42
MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAADEo
ZYiEABP//vexj4FNyAANlzqKeh/hFbH0kFF6sdmgZOoAAAMAAAMAAAMAAAMAXtqyUbSDGtmnhIAA
AAMAA+gACugARUACVp7+w12P4AA0Ddlz7ItF9KA9EgkhKo3VJbCUI3C3Yr8LYQ9MyQItLeJy2x+z
hZPP7WHmhjEKZhdXNJ5M5PVNhLlpG7uOViV9IrZQbJ43a1GmrfmACi0Dp639kaiZIWIgKm3GjcAB
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
dHNzAAAAAAAAAAEAAAABAAABAGN0dHMAAAAAAAAAHgAAAAEAABAAAAAAAQAAGAAAAAABAAAIAAAA
AAEAABAAAAAAAQAAKAAAAAABAAAQAAAAAAEAAAAAAAAAAQAACAAAAAABAAAYAAAAAAEAAAgAAAAA
AQAAGAAAAAABAAAIAAAAAAEAABgAAAAAAQAACAAAAAABAAAoAAAAAAEAABAAAAAAAQAAAAAAAAAB
AAAIAAAAAAEAACgAAAAAAQAAEAAAAAABAAAAAAAAAAEAAAgAAAAAAQAAKAAAAAABAAAQAAAAAAEA
AAAAAAAAAQAACAAAAAABAAAoAAAAAAEAABAAAAAAAQAAAAAAAAABAAAIAAAAABxzdHNjAAAAAAAA
AAEAAAABAAAAHgAAAAEAAACMc3RzegAAAAAAAAAAAAAAHgAAM90AABnJAAAIFgAADzsAABS8AAAH
UAAABsUAAAMVAAAF0AAAAlUAAAVJAAADUgAADVgAAAMeAAAGcAAAA9oAAAKmAAABfwAABNoAAAI8
AAACSwAAAbMAAAbgAAACcwAABBcAAAGkAAAEQAAAAlgAAAFbAAABOwAAABRzdGNvAAAAAAAAAAEA
AAAwAAAAYnVkdGEAAABabWV0YQAAAAAAAAAhaGRscgAAAAAAAAAAbWRpcmFwcGwAAAAAAAAAAAAA
AAAtaWxzdAAAACWpdG9vAAAAHWRhdGEAAAABAAAAAExhdmY1OC40NS4xMDA=
">
  Your browser does not support the video tag.
</video>




    
![png](11-hipoteses_files/11-hipoteses_54_1.png)
    

