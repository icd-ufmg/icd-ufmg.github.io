# -*- coding: utf8

from scipy import stats as ss

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Para evitar a confusão da aula passada, colocando alguns defaults!
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

# Aula 11 - Hipoteses

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


    
![png](11-hipoteses_files/11-hipoteses_4_0.png)
    


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


    
![png](11-hipoteses_files/11-hipoteses_9_0.png)
    


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


```python
#In: 

```

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


    
![png](11-hipoteses_files/11-hipoteses_19_0.png)
    


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


    
![png](11-hipoteses_files/11-hipoteses_22_0.png)
    


Agora compare o TVD nos dados e na amostra aleatória!


```python
#In: 
total_variation(df['1random'], df['pop'])
```




    0.05




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




    array([[0.09, 0.14, 0.12, 0.57, 0.01],
           [0.22, 0.18, 0.06, 0.58, 0.01],
           [0.11, 0.1 , 0.12, 0.6 , 0.01],
           ...,
           [0.18, 0.18, 0.12, 0.48, 0.  ],
           [0.18, 0.17, 0.16, 0.58, 0.03],
           [0.11, 0.22, 0.1 , 0.47, 0.  ]])



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


    
![png](11-hipoteses_files/11-hipoteses_31_0.png)
    



```python
#In: 
np.percentile(all_distances, 97.5)
```




    0.11999999999999998



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




    0.5319258333333341



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


    
![png](11-hipoteses_files/11-hipoteses_49_0.png)
    


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




<video width="864" height="576" controls autoplay loop>
  <source type="video/mp4" src="data:video/mp4;base64,AAAAIGZ0eXBNNFYgAAACAE00ViBpc29taXNvMmF2YzEAAAAIZnJlZQAAreBtZGF0AAACrQYF//+p
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
MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAACzT
ZYiEABH//veIHzLLafk613IR560urR9Q7kZxXqS9/iAAAAMAAAMAAAMA4MFYyqF/5vuT+RSYAAAD
ABuwA2AB1kt/EUNfkACSAK//sUXPe+4hmy4LQAVzQEHxTnF2u1usjWmcAekzhOtPVbSweoiIfP96
H885+hmrY+SSxvMRsl3gsCreR2E20GZETZxrXh3pi7XJ2Zrae+Za/Z8PsJ12d8yx0IT/fVCDcem5
6J5wUFouyFU73CWkS8tV2QN33YSkwMzDTw3/gjbeUVYhAVZ30DWQ6l1oiF3z0OjnolAXwAzs2TJ3
01SecWX5ofQCHjcvV5htZVPzClf3E42hmy9nbbrxAKR1A/lKoHwibbx/zUOprzTRK/aJaFaBjyWM
dgzw2+VyrRnWPmY6bFt+ooEf+4pc7GRLOcfpJklax9aGV3puR9pqKVl+ccsCKtQ2Jwt83TyUWmkx
jDGEgY9YdwEZLl3BGtzRB5mKGIVu+1H9QhHpCIHgrBiKUmprDkX6WinHJwPSnu8kJy/gYWTUJP9F
qjASnsQf2XciAkUlsNg/OvrA1Il78jHkM5BJ1dIDaAqrMJaVAtbQasCEGtdnA+Eqpt72xcE2V78a
iSfi6XS8FRypIQ5d+nnMQ0V+GwreddKEXQppoFqa0Qc7UZrLaIvHOe3jgr9/XhLe61XZqM75jXs9
1/fw6M11I+q7YO5AVBFSq0RmHK1fPr7tY7ux0NBaoQjzaBF4kKFFfpi1EABrdkbeWCz8YM9RS7sV
f5YIoMsncjOJIQKfv8oZ8hIPRUROvUq68aV9heO7w68wMGcy0YuftNdcNpK0cAEPCh6heaaK6+7D
zJmhHyiFz7LyHCNzk54Mac72YH/DYpgm8i++sxgh70WG8KG5aeO8SRFFkj/MHAkDaoeN9ulEMwd7
nVCW4qeDtiIQIFD8JGd7Wno6MfVsViyc6e1QOqwQR4d4WhS9XuAxYdNibyWyT+RWSwIKvBe4hi3I
+/L9oPclr6+edhfk5OJKazw2sJIv/4R8LXdcMlBubzoUzhZIw1TS//QPuW9xAu1adCcYGnvhcVbj
5JeYJycUBp7Bvp2NzNahyRYW2QJbv9LP/W5Pmh3xjf0va5tGEA7xn+lxVFMz7sRSfttDxx4CJwGG
glIkOjnok/Oh4iycTU6uEbUS+GHHH81yP2A6J9bhYBSjxZnZICfhDRxWLHrnpZqO5MmunOY/5ZaD
JSOinjTAVzsnOkYaDJcM9OsYKoMfwKq67F/bq70s6lbKQN3gYUGWYduXqts+iJ/9qzwztC/TzvyN
62bttbGbuBwoSjQ2pEalhpt/VoJpfG0NJPcdCGFVUQWnv5YvU20sf5VHe+Pv/LX6/aWMp8u61RLf
/wpFP2xDqatOsgNzm2IGEetXd6uaiNrqsds1fhbBHSifqQtztm6n43t++gHgVH6ZMRSG+Y5ZySBh
3+ITnCYupc3JWt205etNr0RjPDRv7x5jBL22HPQXBRWHSJgxsyYvzXZ2Lm27suReNjX6qGv8OutY
TLuddOdAnam90eEh28zrp3CDfF0exJpn+3U1Jv+rgkc5kqXLWAMsAdw7opTg+uAcZVIdSGVh1ALL
eIw4Ic2fiwmyOaYEoqObnnuoaMWnlrPtFQD7/cxte4QVkSA9oJsFEUBpSERP6Cj7/i2aHDT/8WJ3
JMbA9vj9tDVbgOfqwmkmcfIqkJca5Ee5vXiQScoFEIhBMKKMyJiT6DrCELGKJeF9LyJz51vI5BRE
9NDrRqGCoLEvOZcnYUk6dQkCd6dnU3P330bv/A1kAPc4Evs2hJfmT7zhzZ9b1U8KHXbznV1cMxlQ
R1g8Soy/wMuSTiwfCe6mVnWxPMcH3yyTpj/wD799w10Bn/sj0HUAOKtWof7FuLo6CgvPHuSzNueV
V/UABWh/E2NJLif1MNkabbzDwWnNHyzPduD5ldEBBxiKOqUiEwcMK4p08FW0BExt+CKXHhlT6PLc
WyrCQzHncEJph9mvBRxLDmTjMIbIgUUR99CS0lNEM/7UQxPr+se554xQAPSpb3pkwah20Lvff+NZ
oNNfRjdRlgBkdfAzlHeSigDNshvQ+vw7OfT9hRJRDhIb770fIaVWbC6puEnPC9GR6oU//vZFdqIV
brzmXomC2U9N/JwXdJ7OiBZ4gx6nS+ep37C2JEoQvVU76HpvYTdRVi3c5h4jRApoMqRztGeOy36X
QfLwefmwHvNf1LHdyHSMayQddvWFW41bPD7W1aIqkF3DwXHUBHInFSwG2s0zg32a2SoLturreBcX
qIFB8B6k57yJEhhTCYKN/64zkdy+z7MNN49/QaBMzeDwO3L4FhY7TPUVSMbQhz4zRzvORj89DgDj
KvQZfvc3x8dNQSori+rcbaJuw4DnRwRWhe9GUKmL9MrEk+ifZ340sYgk80WSsxqnXLQBxIeMypj7
HHQUFLA1dnBUCb2MT7F1whRSS0ac4boNZ8bWzwMuUotgsJ4KFXodfqQ4c1raFDqvjNyilzBugu/S
y6saoNWpnsb0JqisKXA94NurcNP6G1fR+Mv/a7dG2jHN6ppxt1XF7QT6PcTDNhnk/ZIfaZ8/wAag
5rP4gGKfgvUqv8BPuydLhzSAg20FBqZOWHA6dK/nY2sHj4tAyHbVW6EA/7T0aGRIn99S20UAAe0X
XycaZB8exlWlXU8Vs2z3NunptoS8ajvs/bUTC2GQJaFftk5ySelhPsdhYu2Lf3zJHUqqI+es6Nus
zP1nL+GLalMJxfQLpEweN2fpGQ42z2CgRO/fYCZ1FPW9JSD6KsSa4yw4uACYoMEsNUXdDstsmTBs
25d8ipurf+tN+FEbJ0ygynKFAbj8Sltmv3XzGyyNn2u5Uovv5h5ucH9XNa/jcEcSrSpMk3BMoo7k
3gDIhWnCYLAmhm3aIyei5SamVnu0Zy+ItZJZ/hMxVhe/EVbJtdWRQEiVPDBAMrkS9+vWeL0KwzSU
MOe/JkAMNoJcM0B0GwbrMPpAxQ71etqiX4BhX03IOe7rDMOYAQYiFt06OojFtE+woDjGheB/jUwh
7Um0UyABZyUoaW7VTZAocdk4dWo1x2usc9Nk3WtPCOY2X8PDmuwF5wm0SEqUiDnom+LX50QWTtw8
KmFeF19sl4cAZOigAEo////8w2tnd/ywpHAOgIimDxwfcGuTs8qwtwMpn4drpYdNbyPq44Ev2c4p
MLBQHl7+208vawpkMSDPUMTwaRnStHASlQ0aHxYnapghB4RXuuMMBqFMfPK9uBYOI9rgboEN/Olk
f8zgvIC0fNc92M9GmZmfCReuXdk6qO3PkF0Bg8T4hS5hgO6Z1JfpT10QfR32hP/1exJtCOxxUEeF
sCX0hepE3WB9OTggnmdV+pYc4T+FWRetlLDK6TDNmTdIjg5Drdf1py8acSD5gFI/Z2zivlIoWExZ
C5ZFxytZfNGmTUKU49Ej/s2PPcNHtEHyb3JZsr0a6bUAMzku+fauWBUvm1owA/Po9rvbP9nUYCgE
VO9VwqUsg0MZUjTI7VfhkBB/FLTZ335dqUHpHcwB1jaIK6VHXSVZ/LzjQi17k6Ysrdcrzb1NSIlG
KaRNny6xe+W7NOT3KDUPdll+QmkcH/qvKkU8LDWFSXjyhzukStG4jXuf3Ob/DHf4xWD0nBVT21Cy
p8WMD+JaEw1u9kgdH5Fxg0i4w3/uX0+0YjRK6KDWxIREA7lR0KyFOkHZvohzZWX3SiIUHo8roTno
bmhValFqQ9pgbPC2f17TFXa0+23U0jJa7tUTB4rIUI5g0rV/Osw2/7eYO/issajPMHe8MS7f79oU
dxbGZWWjDLdt4/IHPJm/C8gJO8oIDLq64QOALer8u+W1UdulfcGDnccez4eClG+5hNxVtNKbaIJw
wUp9I6tQ7oa0MAGfi1Det+MS72tLvGaGf1EgaX16vFSQyCl5iLubz8bZKZWY8qld7hfLf536J6S2
QgxJie+7I0i9jrQwiJl6NCaHUvuqeHuTBzdiIE0gXyLyffs5d+orS6FP/Jgd86j1TS8HxtKvSy1B
3txC3MYIsEjePhldzsmAdIzLvOJ08AcHmIjYZ/aE1mbzlSxF+mnEZTR/edfRfaryXl9jiNGAJ4d/
Fz5+UznLNmpMvoQZAys1JQKOLtE1KddqoqzM5j/ek3ONks8zTQrRsJX7d1onsmMVINwjz3+rI2af
r6Spqgv8AAQadjQowj+chY4TXpL3UfGa/O6ti72Sq7SYNWR0vwayI0+13RkrH7TdlcinPB4yXktj
As1gGQ9M7zAFvKafirgPFCORbH27SMyuUCQH7KXe0GHgfc1WX3oS10ii/w80vh/Hs2FcCwdiIDtx
vS/MEgwOuZpXPTGalQ4tameGQ3gtrbDUZxxg0vSb2ojyTNISX6M6UbnL0YFrORfkMnRL2tkFOOb7
Fv3pHCf78U1kUrW+u/O+BRVVOYMKcYt7AZ9eAGTa8ZiQ0EvEmEEQ2jbmBem6pyGlsxszVwGq2v3/
TwB9NBdYMkoFpqW+wBMBKN6c25A2y7/ipYVf1Cc0v1bCHCeQmyaIPzPCcsudaIiry+dlh90IKdkh
szAPBHWK4JWd4tlmy+tuN13EUlqbhwesMF3D3WgACkvuN8hKcm6gyzMIC/ybeq2ECQOKfXvdBKNZ
5xRP+nlZP1UF7cN20dwzFYWyF6WGWBOf2x5/Yuw/7Nv1DgfP+0EXFAM8Nt8a80MObgHGRgeG+u8S
ylYmPDYyts8/lFfYia61gq0WxFuFlQQu7wXPCMBSZo0m+UBhRG4cTPmBp/VHhgJCcUYo44lp9qaj
PVR3Z7Qjx/JaP4/rFWbpC8MRa65I6rugMVcrEF7O08frtHvUFXgS9qMdiJZMXcoCRAw074aMJtS/
vfx8r3j0KCsTiwpDFSBpVWE9Eso+L0xSuiCpy74+p4gGMVXVskPRGXDUeWnFytwyO6ypoFY9h5yi
sUbWf8zbKSShmQvZRJrNZ1Jn47bd6vtADMQeAq3xi8h/+oUvzefCV6cuSjd2W84E/v1oJHtgPqvC
AhqlvEiSwUhkfG5ZOQH14wENXoyePdypNBrY3ZkoraWZHWMnlu1hNIBk9Zjw/oJH+sS3udE+zCsB
ulnbI2f1t0mflJARSts0RKUF45xoqnfBPRS8DSewyT5vjG3VPB8Now7JwKZdcL/T7oK+A1EqWqn8
IMPNCmjumi0YJnvufBb77nWLg/J1RyYsHkLRouB9aVU+d/4RUD7wLdWfDqJFHQmGjbGL69GrSA9A
7ot549EtVfk3vgRYuhtnmiX3sTrjQgd+GjNMNplj7ubC7eu73gLW9u2W8jgrrZ42eSUIEaANdS64
m6lDVLytVCJJy1oOIxGymNn7B0A24ZJkjFX8Nzfh+wJCRjHUp3u/Sgqwe4EpUZJAKFfuVl/761HP
gOGERfS4U64AQy67S2D70pO7/6xjiro11Zsz5jrSY/+l84H2DcLH+az75Tflf05in/4su/9NANbL
BxHo19FWR4uT7BxybvK8OPjUzLcIsyEf6PXCb6wKkvkh+2sg5P35KOoR54xsolFF5uf2KON4EFP7
KlEM4WfOAJtLYRoltJx+t7BmRfgWvmKoPvrOaOyvmEJ1wywJ5pkay8ElBf0eKQDw0lglX6974vFW
Yg+RkO0ipUKPV/Garpz5PMt8I/MuebV3pS0N8OrwXsgMzOhJzcCDzU2ujC/5R7TiV1x2pyfZ+D6X
8GSLU9T5olB27JIWSQ5vVN57+JOWdPqTI9bbTUNZgdSt2XxP9RllkAFci2CQCveNZ1HXcmOC7Xhx
/wdM81+9qfJGzzx0qhY/YBD1bvkYSNaml1C45OKrwMENR7Hoc6+vtdKDZPg5TzCn+OmYQxyo7fb8
VjbvvsZdqpq5Mz3QOxHqYhHvclWksUA2C/pdRMlDBpSyWc1KbfrmvJaSQYn8WzsTwU92lex7cPE7
b+Tc1jptzqb/esvJ2KbOzn7a9AWRc12fI0TjU6NTYveWc4GUoLQ5iY6WT4RCY0UV6p/9Dbykzg5k
2GfYR8wo+uLzRQwrmqCw7vidMB5KuTmVYlRYi0txU5+iDUQ7UdEQQhfC+H4NxnPpIcnVmwwb0j+p
4McdEGZexsoDZSECS2V0dkB0pyCKRHUUqKSriUXxIPKoEmaT4CfTLXuWUt9d0w7MTwQ/9EEI7V5b
6TuLBbnXq1ZZ4Zuw3nId5xOF9qAH9ue+EnX9mTdLFTn3g1dpwTwcP4r0ne9Jr6b7kTRSNhSqrKcm
1uJh0tS/v8ID5WfTQnfZ8WTiCP0gDOlNYkLQKuW0O0PIAvUQduMeAAy4BJra4MnaNVP9FbLY1qP2
eAugnm1LN0wg+I51ah8IdAxe6AHkarTyUzKE33pkbriRrMvwcT3HfO8bus+UllXJwWhocDKJaJV3
jw13dvBnSrWru7FWL3JuVEdtClYnfYY9Y5ck4ADMJ9g9HNAswC0wdRSW9cYA1jFGZxqWjrEvc+sZ
JQeN5VLW48OrOrPYzeu6wUsidcfM7nhvAd4dpmm5FkzuIP0gK4siC0B61pDMmdLw5Hb8bAlXYapT
CBxcmdcb7Cmf1BHPCFjzruIgPf+oMgsXpX0SoGubh46bPjPdU8X2irTiH622GkwLtTRXDMd+5Qja
q7URmyCyx4cOWD2Vl5ddWzN7u6h98PjKlFkchdQetxbMh34j4Rk+6+7DZwu9uAURcCzBCD6R4TYe
efnJxJ7Hg4nBrUZdlqSITrq6C9lpaZZ0l+hWp/cPPs6ax52h9toFkFjzF5FTuoz90226R+tzGw18
CIm1jWRyxa9FGaLdYw3u9J7i1A77MaQz27wnauhGW5B6OGiF6kLf2bVz/8Ayo39tilCvlfQGzlVh
H0QDHAyIePrUdCiklYYlJ3UuH9Nymh0r/UwGB/4l02XN39+URa5akyUDWKxF3U/9bKgxihSrL0wZ
wD32QMnUV5euJW3CHm0COHcw+0f7XC1R/KFOGlAOLSYi5IhzrXSIMXj5PIGVg5ZN1ZBxQItW5ZHV
oq8jbmbz2ENdRZ9xcKN2J8PLAXBDe5uu41hrtwGlp/xte+wKNN7gMLGzQ+WZAYwT1hTNcwNQflX2
DAxuEgG7BQGr7UxAvTEJ1znq/rxF2hPs3u+3g0A809TgbNQhVBqQTfR8OZD8FqOWZv2SrADgsuOd
DV8dJ6fO1XTVg1/10oNd8j7n1Wji/cy5rUlS/ef7QmXLsVzBI2PugEn/thTTwbPL5PcvtfOcYSVc
aGHGATsQWYSfkPsDbCrsq7nskNfMPW+9pwHZIr83/L/ixAQYpvMtwFEG3pcEW1WrYDIKaywhekq7
Mv8LgWZvFv+H9Aj4ZQPrfVwl/c3QnQJzLxiu7hWMj737DRabrGK3poK+ZpSKbO+fSrZq1RIHHIsp
1nXAnR5YFNLiqHNctMnSRFRTe6EcpfzGpBaUaflrJe3mMZdDCX6qdaczLI8KbI3/6ybv/MKJMBNd
X3Uevgjpu3DAhRZDfHcz/JhP2oAXIStaCbB5/zX1f+x0GtWdlAfH0YsxkHFsMM0XA9MwFMWjw318
f5l58PtL4xITbIuHp0tu2hP8tnmcScJI2KfXjGASK5PI9uJE9BicaYMpqWsjdx01NKSHDq97z/Py
e2gVGKkqRMjs/SnYrt8RG4nw2Sx/FCVQ/pazyu1E8QgwR80czWbEADNkJvI5sUqpzAjnFN1Gg+5V
VTHvKu1I/3nD/Wy7kPjQStXncE3S6nb7Hzp/MR1Hf/SgHITD+OwZR/75RTrPFB5kI9uKN7DaeTBb
y2lILgGlyuXGLDdBpPcFKgB+Kz7hT4pW6nGGk+0iJ6tO1Y76YWrf72DiRznLppEwpM2FJNPp8EwG
pOGMtWE2OZL90tbp9UeBerMxvsyOyoyTHwbmphwrJom9L3rW524edGH56eMRqVLUEP7ZTz/pKH8g
Cty3nrXj6FxJWLGWkgvN8MdCc0gVYz5g7FxRGjok6khxeBi6hYTuIsLUhk7oYGduHtgNARhtas/m
FbnkiXJNQhRl1b+q53Xm8oJDiHMUzAw4sSaoHYgfib/eOatssGkMwfIt3YwowBOy/ms9jksPu5Hz
8GkfpwKcsXn2WUY0gR5qXtuJGjUPEVH844NLnZPyoqYbX0cK9Y5f4WXoF+syanUb4/BiZ35Gbepk
ud6G41i53+jLOX2evI1NIqCQrt8sZdDWsbCixNNKrYPJ9SfQbBtRKqrKr2AXpnueY6A0ax7E6nn/
DnlGHOLhzpubTuXSUbhbm5PlqCmppy4V/tAVH7qIX+uoou4awc49sArqdMILN4p03i7dFn2thCsF
4/X8IRiwdrRwstibFQXtNcZtPF2rqJ4NW+Mbvz/O9C0jHCMKvs9SBlg0ZuD1utuIVVHcFewQb2hr
wiBfYaN8GEhmHXWdPXMYTWVqxh52wN5Cg4sQmVoGkarDoHCUGZcuMs1PIyRmZQ80EVud1dz8GDv/
ZsGcNFeJDNIeL9MJHaARVyderkeEJaMCLPXvD1MCCTMgxn/tAipgcEUXGCE1oHaOW96gdoAngiCL
aIbH4KGR8ukPSRqXNmqnv3ZRjOnaMvqLERTxXAlf6pLpNj7Ec5S4o1WG7CqSUg+/DlotIftYTk6m
nZfWyoBtA3JnIxG6kegvKdag7XKZJnljXsTwOjbXVJ4MtwBjnBPdtvtLYetj67VHbTeYqAYGgdpW
NwU9rqAyN1pyZhms3+IGvO0U92YnC9vvEzQRb8OdkDHaMHZKXwx3PIH0SeNGJOvRWVmNU60gNSlC
JgrvsbZ5W+IKhn6/SNjxeXOYPafpZOCKhxkaNpk4JPG7TuZcRcMtDZqq0IzpvwHgX2ENVXLlVWca
SzvFaYIoEOvmeGteVGDXH3oRJN8cDATP85JNWvpYBfJKbL8HnaSu/RBDSQTBeNGhY0Rhi6EPCfW8
0OFuhmzl0ARPVYMLomIjwi/QX4V3/cgBNTP8zV13Ke239aCkrv43UmVtnlt/rII7JI22fJPfd7U1
xT6kQKJuc2kDxQ5CPe8E+aAU7AkJ3xdJjcyaywl04AtCSmKneqjpusn+PBkv1d9qAg9eshXOVr+9
aUL/Up/jashbQh4YPBmAX0ck2DZrXxBQeOa6Dn0IU517xIZM+8qARG+lZ+uFvDMthd7/u+ZiHINQ
22VxCGsyvYUtA9GXLjn1sKHg+FWnq4mD7tbgxNfjcVrwyvPrPrFYfQQvR78q1vKp6PPcrWSFeFVk
O+fAB8PRR1qkEoHYtxkzy/g4LLIMWtj0CUfenVZhwhi6GunxawEAA580wH0SaQtaqbysWeqq6mw3
ReKxf5QEN5rNBjqLNlblAIHBBYLA/W7kgMmjmP8BFlXFs2A8106m7sUhtP5fsSbeVCvo1x2q2iNe
CPAFnzBbvqq5P0H8xv9Y0owz9x1qNOxAE4MZt1xTRBtlUxS6ZILDkfCf1pcKutTbBNh0wo67dyvV
3ktR/2+BsfYjeOKxhv8DysC3Kh1vlBza7w+VtVyPUWVqIMMfKOq4MBCJ8f2esFeXys0Gu2I5EY1Y
QmotOKl+hTIRLzHTsNQphxqF845nb+xXPORG/LGxf1ho6p3mLHry04pcvWAJwCf1hIvEwJ/POzTV
gl9AXiASffsnxYXrdHoznrM/8R01GRS6LDr3FttebX3k/1i9GVBsyXKFvfrWYZv+jAtCZmooq+Ds
GEcoPQzQtd1ICYY5gW2dOvbPoypudfGYG+yDEOzMMNfXnjXK34BRT9yf/idG2rv2UYVweX6mIk4D
HVlnG4Z9lXnPrMcoYlFQ+OGUsYFRvl/vXU8LU+kFd3WwP+jB6vGMRQWL7HXzPDWvKi/ahR/xX2Za
x4sv6nMcd6sg5GfIvWk9KhWf9Iou9FP2B2b5xXZ5BEIQXTITSPvXrIytPBviS08holZAxamEkmyA
03q172RdcXnJBNDD83rDR2NvDa5Fu2IZL6UgV/Fg0LVLomgUHPlgNeJkRYAxCZJOxfvbyeNfMrkV
NqYyILMY92RCa2QWJ/jh1P+9pPr5y2ZV//R/KQNg6UR/5zzg6tfZnKAk3quGzF7rHzAtWEbQOepR
Y2DhcQ3khlF+X/dQ1p1QbcIC3rTUh+DVWqRvY1lh19lrLGINMucn5f+/z/MUDbH2FAbIy/ZkzlCv
TSs66YI5k/39tndcLgD7jruyxPQIoi40gvhl2bFqYtCrgZuUDTw/FYmzDbC6i49Ijm+TXoNlb1B8
5hyHSbBg5ImXlLJwdbdoZ1zmAF9uv3azByTOI14z2Pjqaoq8oMb1Z5+ghl2Cdoi/t11r53EX/yn8
lQ97GkBWUwF3p5vUHZukWOk+XnykIXFgK3Mgjr//eHDKcq3r7yFSPkAoRNV9Z6KenxT2O1N1j05l
4XizIYBAv3lJGz2sxh5JKSnU+41KXZwsClna0/hQ3iVnQfM+7xLm7PJJ+ouwvFoX6wa0U0iVNpPy
+2+zERdbEZguDiAf+qMoETSDgJN5DGq2Xe/iXtjMPXGEw335ejPdqSiuxy47JxO41pWzOx7KN2w2
y2fqvxeuf++PE0oz0t52fI7kW2/VFmN/xg9R1EGha2vxm3F2kvlrJncrbb/fT9SmD4ORJN48iCNr
WB7sEtjsgsv6ikLmvfIv1H2jK4CgE4cBhe9LBprKu+YTrTclOCDAL4Oy2afme+KF5utNFDhhzkUG
1CpKmCjU0NXRJyBUIQqVKAOJl0byX4e3RDXkIcMTPIhvk21+qmsIt54C63G2L6UBKPoVieh44cTN
RW/TrzcG2Vs2vGRHWSyKcr1wpE7+mn0Pu5F7/ZvwsPuby+ShvSKmRksXtHpaiDXierELze/rh1OA
GM7kW2qDeaABT0E1qEYTq9ttfiOAzUyGWfp/my0cyMclJPVTWnObf6usBs4VCaVVCFGZloi8bs2Q
/VmpQDIfZrbmdv2wPGHjj25mWTjMrTYNaMA+Y8F7+DxUWnJItl0Ws5Q6ocPvxTbRq3zDeHswAQEC
CYFXi3boiYLeXO87piTWWF1eUNsDc5UvX3xOsA4mOi7FI0E0edn7hmEXYKHSr64mqbCdaIE0OZvG
ZkwV8gARbO7Yh6nTv2wrzUiPW+vaN7T4oSFqtLD7O5uILce+HGHkhl7DA5aMW9lxGPzbaOgwdauC
RNbjMPyf/PrBp+w9Zv1BE9vMF61R38edGdixAl+D+WhazdgACBCYq3wYBqAYAP5JgRfdHtlLTKaO
Ft/MC8usVTu6KoUiZ84BQufxNVIisENuHs//+4F2KtQmBhQKXc0lN8zLnzlr4f/+dnrveMw6ybsu
1FxeJPc99HGHdRm3SkNYDzvGgFQr3/W1j9f/i7guQYGn+z94kUHeaT8ErTz20bfYnOnw8OHhEsaL
rnx+gzy8SOXZkvtlpCSHjb/EC1xZkOuFjencl+plkmmtKCpcXN5NhJcf/01ziwZE/MzVpu0+3g4G
NXR/ykh7lW14GcNrsN7Qxq0eH/pyKgUXup6kJcr6ByXu49vIkXprtJkUD4Qvo5Y15HjB6IemUjst
VTHowhPWy6DxzkADyniVKEHXBA9BR4aHYjQVHdqDDwu4X+ZeOZkuHNOlSQxULLQm4EP5PvyDRiCG
Zybs+Mi10oKza3T/won+5a3NNbsgp1Sa+H4sIW+YhSHC5YYD/5F6VpcseDzlbGtN8yUGvV0e7VSN
+WdNz9iSBu4lgTdrD0wPyx6GExNlTTQhAwtUevpf4mX83vTLwT7wOY/dgW8bsM3NbNPH3jXCAR9T
Ba7mV7EjToM24ZQVsW6QYiydDL9Mp1KecHjt6MvdpUuKktZExfAUX4YfrmDIl6lpon1yc9l9v5cU
fORscb3Kt3sI7bAMR2A2Fo6Y6+Re0x+VJoBd0PTQcaNV+4Qq/W2ElaFcOjA501kIZkSKsmrX0lBD
QKmbczg5BAq2YN0/RgjZhaVLenVmx+lFhXokgfLawPV62p9nRMivhxXk8BRHqfQX81+2iIwCSLXo
nQAcW4kIts7W4dsXEQnYLnjqoiCSaGUNRoITCq5678PCjJbjuUDMcCkTGkwd7Chlmw+vH9rclXIT
2xHeB1h7GfglHWTCJcP5b4VHnQqatgTKkNnBzXvwWWH7+nUanqDb/rgzEekUEDS3TW+Oonz1o4cJ
/Fhw9ZQswMwwy4fbhjY7DH9Q40Ws+Z2shLYYkOHONJBOAVoBWos0zG6Ex8fNGloXgkff9ZgEE5/m
WVFM7loIZU+2wdz9coJzmfyMBy+2sTXyVVMGFhFcgq4xqgJS2j3wsSR9q7mk+DsVdI7zP//wRMMA
AA/vJe1Ij2OX9x0bvX0Z3jpoIWys4BkHQd5N41eiRZ4TevNNe+LwLN6MgMOVWPf5yL1Xobdi+J93
Zf1Z2kon3jKtfid9kxa6jZTVZnSL3Z4OCK2CSsJu63H8e1ZiCz7Re4+GARbWWlvxbKJDZNFJYkI+
NfMvSzaadIspxc6kb0OO0eqrVj/TbdEA3fcQ4hCGe13RUyJNvQW3ZkOnEDGiOUaa2JyV9VHiYSQl
0A4ykYngnf/in4EoLBHg98n8VtfVbELyWZTzQPSwfZsPxAi5CEWFtajXvHHRRROe67IV1vFOYczV
kEqljj9Jgshp7w3lqYq25ItiQEoOa+Q58hRKEMPTuvPlnEZSQz/5/khkY4BBkwaE2v2kbtgVxb/X
kJ1E5rfnBBZObqCNyrgSafCRhLNBUJAh/eYqYZAcTQfgy6lBTVKGp1uxgxY2HF1tyY0h5+mhvynu
6167bm4IEdOvPLRCxOdyaJwgU0UmfKuvDv7BE5Bv/b5KzhjmuY4RiPdaOnRQAOMYeRYCyPAzC04V
qoov90VMLe/lW7RWG6jZxSDNEiYqRmZEM5E8fUP0S+UQy50Ifzr7YyL7C7W7QbNGeOKuXLEGoSOW
Bylhxt9mlnlsjwEWGflqP/1ReqZvb+6YYvCOQRhYw9wlO8PX9Pg/Fb/uKuj7LllOd6HY3Fy4y+bl
PRMq37N/B04uzUlV+/5rkLAFqUUclC32SZCUQpft/LkpnjE95uFzTIxuTabjqeTtXKM+9ySARdYI
4CISeZ7CO5aD/dz6aGrk16V9iE8l7O685RjfKrui1x4RhoiPX3lYblGucGX8D2Mv3T+tRa6Cf42R
u8tGODfBtxD6vllrRzmbnKZieoVtSNd35idfE34kJsgMazNKSneWOcxoRerlbyeCqMtfHPLDWV+H
MUBA1/jj8KEfZ74R2tqoVj1kXua8MZRaDNTeyYpPYBaEyvIG+0mVnT8h1iYy4/D9ABgTReH3eEsi
NcwiR3P7+UOiMynTXYX2PP8rne5656v9nwJ/UeT//4tTb++bLt1UBiRq9q9Pz0yOtofF87tGQd6x
NvG5eEVF7+PxvJhYEH8EArtaZ1H5rUE6W2dNG6qFKHcP78aCD6kKHr9WZUq0no01mbV1XmgFqesd
NmZ+kEme3w/q3XXxX2ufVmhubBh/D/w4kN3aVeOIoRZBWHAUwE/fj365lTvkQfoiodvyjoQMnRVI
Dq3suej7sbfGEMeOjnWfriHtyMn1GpDJ/6e2UUZKgH9TFwjeRGPOEVguFOuYs2V5oT7Dmw6t+xap
9yvKcEk9ktpSUiDFwUlMAxchb7XF2AQXhlZkKF+/p6SP8SjkiVzsd97xvPzruHDyOya0FTmzg8Ep
FuyXgEUQUDPz+6mqL2x8prXewIoESYscuTQi/xceQe2dHAhtDQmYo1Wrb65Lhgg/SJesuNDm6Gk9
imkozM/cT/27jFTsvKo4OLdZN7N/rg0J3U3U1xoKeTYN2OxwDBtS3C8C/aO+unLd5NOGjMTwQHJ4
lmgbLokqp2Qp0+UBIQNWFDY175d0+p4vNeF80fNkR80mYY0OLUn/b3JBRHvwIJKA3H9lUxKW/ro7
8v2XdfM54AiDP4jrzMQmt4XBLQQzIRtPLt3aEfy7l3S1foMLBx8rY5r66JbhClg63B3UPlf8T9JP
SeV0MlYT6SQ/gsDPppKUiFMBBAAHnk9l/P07fezlsNE7JeuV0z1CgJwBRc/pKibtmfQO6PitK4IT
322XB1HcwlKTxnEN/IwZtNQrsEB+mzNMve6oxxF1SwvFiGanctgSllU9oRLK8GuDuO0ehD3t3vOj
s7MOS0l7HWlFqrr/90UpC+iK9yugPF6dTCGAO4O0u3BzE9A+6So45UuP44Y6fzSP7XxH7SWqc+sS
Py+ZU8vwk9rGyk/z8Qj9zTmc2jlg/0NtVNd6dynrBkm1kc6S2sWiPc1D41oSQHa5L6sjzcJxvnCK
Mk1KI0madJ63TJmUJNz7H267AE5WF1jwlzJ352tv6BKToumHMhq2A9N4YIkhoK66y4xk3ogZVRgt
8UZ0ZF+xpTP2ugO7WXBMw6ObLx6TUd967caqbhXh9hdMQxGypsV2GwI2EEOWvBYR2BHWl5uRm9Z/
YVFleklgV6YDCSwrZdWBMKzsluK7ID3F4gzFKfX53M+O3s4MDpjPD1yEf9HeU680cT33TkdgE6qP
p9Xra942JOVDGr/39EP1Ks46UKUseuw1Dh5sfOOadpdR5oRJqUxWHmGXcjClEiZY9dhqHDzY/b6G
AwC0t9Xobn1iqCv+ZbVAWzrZ2Pfp46915dyGJ++G0mxujVfXOxHifFvKeWqUvTcqz2AQuQsMFkf3
8zRUHVokMxZ1yFMs+4qGCJVfNiQMBxPT+UY+DbI60L3FnkEOQ8kSjlZn6Aes7oKmmsIOZCaEqFiD
ydfE7N5SJdZnCN/4CNlWOok+7EeAy18U/oqR34U0p22XfHXg6Ai4I7uSZcJVk9rSFh5h6ZExpIxm
zqXd68A5jsaHR1OVL2paY/AJHcdkViphJ0S6hozIGw9BoP6DQIzfI24gP99d7fp5ObRSMf6dg9gK
DojalGtJp9L2HrFvSmKSuFKpPbOR2zGrAlYo//7eu3AJG3bx39f+oePfNYCKjy4RWIOB20n++LVd
eIEad/NVNrAhEGHJeZJG7h+vZcyiC0PBs2eaGTGD9YY0lm/6uEyWv6rpMBEorLXFoeBy2D6//zLl
GqJIHQCF5S2l9yLgiA3qnOHRMqKOfAEX9RrQ218e3FfYXdV02WBBIXSnsqbsx2P++0PswNLfygcK
c6xn8GpF741d6VBqndd0QvVtILxVGqgBkBmMWsi4tUXRcj4pMzRqfpiQqzLJZ85L8NTx1J0O1C4w
CTkKlx7roSooETv2kLEwBwLuqo0xW9nZZkuSQKFbfrAyJ5HdF2FYTwps5jbXmttD2Via1nhDdvPr
H012xPYwR/KZ+jRuXgne3mD8duAB/TwngJ9L2HrFvGyHKDBuMbmvkU5szBnkWhQROe2bItXFsL1B
8YY5J8/FWVzixB4eGAACIggJAAAXbkGaImxBD/6qVQAA6TMIAA43mLoXuMHBUylR1WuEPQf3mNEb
/ufS3Cu1HslgBCH/rYSSeGvYH2CKmpV1488BumGFSAztXzud9knyOmlcR3XfftrHDFBVCpdTiFu8
/SPChA1R/MZEuYTHD+CYPGbN4MFuzNVP1pXZK269Zgt5PVTBJ3wEssGKo8tMwON+yr6ex0CTXYmV
+dE+ABUxrYwlStCXi8Cp6kMKMqI0zbTfpcYKtbUieJk9fxOYwcTGoDL4kVI+BsfhVYCH4RWQ7EEN
duj1ZhQ6AnHcp1YytMbuWtaAEBzBUGzs6XFKMCXR7uuoidJmqb91ctoJnCzf67XkCFxpqo9jvPTs
uqT/tQ3NpFKWvSrwcI2TkBasCyM+Z1Aoy1tuMDXCfj1fY9353WDYggtk5F83cme5I2T0yBu/FZWN
q52/LiOwJ5Zt8jnVPFAU6kaLVx1Bq0y0oZEbDNlQimmF0MiukoJ0MR2nCMPo35C12KW2gga57fQL
MlcHM21vRqcVgqVKh7UM7VDfw6uqii90QhQkxHtHpm3MB1u0Cde1v/ZD3uFzrzL1ZDQ7tasv1T5h
QT65JQjYD7AveIC56PaDA5DULNBFFq7FU8GqiwzFZmf9gsNpDWSlWwku1Ftr3obmu2pym2qcezQs
vj5Jq1FcDUifXErb52bnFGiF/jHGcfEc/DuoT72SCP1epW8Bcr2AOW2UbpbTHYpo7q7vGLXkixll
+XyXWgZrpZbCFOxqv9BL3IdydYNUQVUCFVv2IKfpebtp/VV5SBnfhXvcpeYb3yrm88V0n4qGA6SN
klnvT3n2vpi4Me9ug195zI504r2fV66Wz1AAQE/vL639PRBy7+WrqQB/BlkcRqyrNppsAhjwTA+g
ybcM3ZHMt3xwybS6w5Z2/gzwCr2jJ0d/PYiIw8695Pujpel3AAP8rlkm5J3hdSAzulPzCwZvZAwC
8W+B+r5Ef8QvZOOFzC41GAOG1suXzD45zSIs0R+ttgDDa7+xPORGIkhF//9GOsc+ctRBfRy7vnBK
di6fXLi7nHQMQCpJTCIgS/oddU8SRkxB/2gCjZZQppqmw+1vpKUq35C5XJQhU1cpRfzWaelaLnIM
x+3ZkDhPSblD/DWESNcyMS0UhWP70Ok3nvKdIhITIHwoTtB42RQJqPiiTebR3qIudhZUgIR+gNg5
Ht4QQjP8e3FCLX2le9ttm8WexQO4RBsGaNv9zBuhEBIU+StqwDg7albzZm5wtaWdtZ/GxV5WVNpy
JZmdZwHhzqHJ10jspzseuvyZrSfKuAw3RLkxsILfPBDGgUpu3bQ9g9gfR9kJ3Mxc2LCIUfL309BG
F5F9tI8cU9BNeqyYWyX8JbTuO5Thr2Pzjq09wYAMPcHFQ9jePXZmIjVE6EOF7AMvsyAgKrzGKwla
bdQ6YjI7MFPNhovrrrdexIXxvwDlk02rGtfeljudXK5s32Mws4W3C9qwkC+BoNDI+9zM2OmcHU0D
A1wUriO7NxejLUuwIgauF/4nM39FLZqGLTR706C2fYvYZb+Ycer9ULY/X+VW4aPj0/LFC7WZL7Qm
EJJjexmbsVh50sKnHT1ZeQVKS2YhQvFQfW+qhCtJ8lEWDddAD+soq+iKJgwrC+ZmCHlHtBJrcUj/
t3XsqbWxmLZXQDRntPw5dMtGTHcl+woZNP5nArKGRXd9sMZ2BvsAdgdShqt9rkC0ZKoaig2k83nO
vGq37FoUe6G8eF87XASW8EEa1H6jpivpidH5s7N4CTLMWui2ZksFT7J4dCuAcDrBbB7fbr9c/DOH
b4ailbl7dzlnKKO42AFzZpGr5j3aV5Udy2uEMGW5JJcMDiEaU6/KmFNhwI/S57nP0fHLM/MM/NPH
MvU0ujS23FgH9AR8U7i62tEPhiL3Jz4pUtzxj8OF5MVMqS8Hn+dohpbkKgZF1ZA05VRw7JTvlTPW
MF9RU4/hHAw0++rFtAMUMs6mZg7UwTZOjlsaB48lZXg9Rv5ktCPAE0e8T82fQmCtfGmhmkOi2j+e
vSHe7M6ML5e+mIBzJn0z7fQx5+Q4TyJjc/ZakZzSMZAWOCThESIBDlgmPFylcYp8QhAZGhLzBMdt
V5Po7ILIKES7RntIYq0xHr18PdQsWyRXFRB9Cqu5sNvAw12tmCxuPVSBd5dtPKEdmrXt/gDNj+v3
DyBPrtMKMR+TpCiKHMgBj+jr7/qAvgE9mR/yXQ5Mt9kEQM9BixfpvHnQM9wmw+HSxBxwoE3S8+Nv
IC5NkVuek8bGnivHq/XUPJYjlhygaONrDo3OW0SsYnWqIoGpmTCkUt/OP1YzZ3dlP62TQVoNJdNK
zeIImyEIsVFkghqaQzT4voQH9QCXSMXQMiLAeLdwu/7z6/jaQjZvUC9GglcCyopHhPpFK1Z15hoL
sMmSrbjlg3GJfttonb4COm/xK35kgDr67m1WmUZpbvVm0EFoinNZlxi6FVnNScWZN+3cwfuYelhs
8QF9hxqEOPOwhECT0BxBiurDtAw3pjSsUqgmDOCSqnxn15j4Ns1OHEyO88YtHhQOIcLjArfLaSbn
kta1JUQhOxTVjn1TnQmqMhnSc4/QycPeZzk8pjd2J8U0sYC7Brb9i1UID1qM/t1FHY8EAimLtRJn
zh+usDVCW8yCFW5tDp7/wFUzAxCtUqYl+Ot4/TIFBYrv1GEZMm8CnAyKKJiPs42D+Zg4hKUv1tJq
7004nacPfByBQnfYFz1/okznZh+wcksbq7SbiwQtyCmARnIAm242Sm6DP3d5hRHLrtBfTX2iWDCl
AHqXUH0/EprkAiKZaRX/mB7GSS58HS57q6wmEx5lPQesZ2dr6Dc8nHvfMQhXFcvO0URT7PW/Zjyb
vYsGwNU7c2rLc8WCqJjaqMd4TX/wm39ClRusB2H68t3o6XXIc0uzKxO+WbPpNHOSttnomhlkVIgU
r7QMQ83ik3GXV+FH7piyuDUQfK5A61Z1BxnADg6aVJ6PjNJ2+YCDJ/4U/lbghDWpIP5B5X1QRqyE
IsLMMUD9e7+v+yBbC+eVDI6peC2gWfhhSMcdWuD/iT2AYXG/dKNOR5AWbYVidbQsHjk4qw6Ef5/r
0fpoDQFfZ+5UOzDw9+YvxiXjfa7YeTmDUp+JzqXj/w//b/OcX2d1BgcL0oHhBxXlum9QxNnm5gxj
4lG6kCLkF8yYOgjZJdTJWU1Sp7XYIEKoUXy8l/DOUmOe0jjSJnFNcIVX9aC3HOA3WArmfsHFJLeS
uab0RMEbzjV1ipg27cf4uPChBSIJQU1nhh4NG6fApD02A0c73Zj65J/Ycot+Q93L0Tm/wLewmILq
UkXZ5HqE6HhUFbdTVmxRnh4G5+NotH/osiZWw1cKx28IQw9wZmKIkE++5EafaOplTYRVMNymRckr
b8EfoDv1snOmeZEwHBQMFPcaqTFiVWANRYAe///8m/U/JuLoru27uOodGs3wGPXQCOSC2aSS7pwe
a2fZbjSToLN4PaYX425BZMuvSIpjoDsWOFICOo3DJcwUoKgtXgOh38t8/TiGBhrle48LioednJPk
ezZkDKMQxJnaiFVRkP0yUNIhjKM+oZ772JRDr/6ApKQwQ50pg9bNXp8fqMRzL4SYC2YTFEjeOsRw
W5+gUGkuIBST/MH5B/7MqL9HgFPnAJIZ4DHWLzsQR97M7PcJJRbC4ShvRVlR9od/d94erfXqDQrh
vWrj2rAtYR3iQXsHO19Oi0GzgjgnDVvbOK2/+SrRJTGd94UmSNxr3GeYyuVnMOhSU3tWz3JgDQln
6KmSeBZdPfJ8AomIWZ45u/VgASY4WGPdl3SmfCHoWMjPns/wR7UM3cDjgAtSd5iEc5n2NiaVtSdM
1k2deAJbTeND+uQGhlN/ZBagq/gyNjARFWmGS981aU7yDSSqVThhGi74bJUyI1Up+fS3DiVeYhyL
d9HRQ+Y+f/ldgAByhU63QZmac2fZO4zs7lM+QZHmolVrFTii3VT2Qra0YZ8B2wp95FGrMDJgRNL3
vAAv9YNlSWZDyOKKgkjD4xV1wNj17KRajvcKQ9ZfchZhQg7n37302GL5AMR9JiZWNfY0tiYw6F7U
6agPdxKyVVQRmcygo8uegs0rggW8O4GoYQmDe900tfhM/rkihvaARH8oR+aHj/IY9RbOeh75/Lym
/3xHmuBwe7MN/g6gqUtb8eLKYXkxkSq5p+XTTePttptbXgiQMu3zFTOmAaSgRr9FRYi+fJhRGap9
nBhYdkGWu7CxfzRz6S2N2ZMa0s39qscfwmFhQ3xbaXdtjy2E5PXLN7YW4hjRGtN31PWOvzYH88m7
Q68pJse7PnCqnV1c9/XftLu6jPyCPpapCqk9SFQqyzdnjtB/3iDBLm+ueNKwryMNy1SdQuTuULZA
E0uVYEl/1Fhh/gSNPS+BAADdF1ZM5C0MhGJBJprJ40sBCIQulQFMikeGRIuBJt9FOPYWqAWGBejB
f55UwhtTVeD5P9bWvCxZa4X34W2vURH58jAAJAayGxooOLqjy3XNENvpTeSIoiUKv1HnskoyMnwT
I/LppdsLTJb2IBTkK9BBsiXtSAXfVRcudc5EvE84DJTi2/WFYg0n7e3FFaJLjvgl/8XPUCCsqlCR
9O3gV0MFhcWI+EMgQhVv65EIGmEcaboYLfowU/bKOZ12J9Vhsr/formzNtgUAQJkfOjmO6CKdEmW
wF0srxMtNwSeFbL4DzAhtnYlhMjqfe/3Iegxt88X1naaAdfy84BdCg9wCTjoFvoavWR8SuAbYWBb
ochXUi5BfPt7udvDce9kZiauvUgrwsdb5Lr+NXKB0IHBdOU709akV5WMJc5l7AJSeBaUslC/+mIB
ls8p8pq28cxbHXE6YhJSF4NawEV41EX6y3yq+Yxdq2qXHtKBbXYaGzSM558oyHKH84nQc+htYXb5
XJhjMJrgAKxOnPpH8sNBrjhnfH6UUGd/2cikBbH9/Kos0N5yhyFAybElhBi0HSbRdhkEFp6zJQv4
BYrexPPL+u7D8zyZPrY5KaTepwpuxJp1bsoYxpUY3Eg6ZH6FefsW4b1qZ+ZtSUG1IyP/iUKSlseW
+nMvqiIIWMblSlOnoXy2poDq4FbjcJVumi4WBuV79/cJ4IeRW/VNLXa5sRcPMtn7oOLX6mi+r1Bf
Klsd7558NC1I0PhIXUJjLBhFffzezw2up+1RZ9qnqUZPVbucj7vL5Oc+ziCaDp1S5IytlS083G0B
BjP+VEpXGuF6HGeafwVpWLkvWlWksyD1vTRpMeIqGjv31F4YYMT7h4NHZmGhO2W0do2JzEdfBh1/
rQj25Syc6qIqxzdI638aAQpGqpoFeZyerfgbhds1/6QzyGE7wyH5uS9hfwmLPG/3q7CIoDxIMuio
Wi/4NnYyOT9AyySf/tzRYJU5qqzsbuV2WcvmYOmjxRM0rjNbW7CVN6RpcsVaiJX6rI5ciaLlobgw
z2Ju5eardc8BDTIoxXC2O4q6bXoV5+CrKyxT8ST6pd6rMFBOfohHxpq3Qx0rrs2YIju4+i19a9Ur
iRsSSLMXyWo5V8B0AXu/DszbRD33KXJuq98WuK1OQZLJKUFKh5zuJrSC6XXLNNK5o0XTAKWIOyF6
hUc7Peb27m6aa8tycajNTJN0YE+nLQCvIjzsgBWPo6kx6Fhpr+2mLGHtrU+j64rx7BTyIHIGQmt4
QauqwLGeULz+xMGjZqjhUs/ALtBUmvpoSlGSBRl7Raau4Gm3PgzqwIOcislFbB49/YIxKp8j/9OG
arABQeBcCmqN9XYuxLGov11uMbF7uDUKGy+42yS/XmYDmaLLOW+xdHF//VhVN42eguYIrzkjFJJV
9GdSF+/m+frIwjlqnHqqqfddeGNTmoBVFYRPcHmXjznZdRA6weGVTbvXDwIOsbIQuKdnn31slxWQ
PhZPonTd54GKY/l8hhBYnU3eX64M1MD5OEKSz4+h7lUW644wi0in82IqCV5pxpvikyStfaDnI2JP
t0qR1/Uixn6Yi8HG/BpEW16AfItp6wugor11vIkCSqRAfrAngu6M+0UTIQT1cgXo636AnturXkkL
oE2aJCy5Gxl3bDnbYuifk53PbMnIuoxzRQi/YaKXfLNBuhcpxQDEMh0O8scIuYVgfBnVrwebsur4
T2zXJ1xO38MLSjoxJoxgh6DMmPNusV09TZwDV8O68MXxEbH05DtS2dPxzEBO/7P/83M1ik7GGXwt
1LJbSiC2Bu/90/FsHqcmCbB79m0jHQAvRqomsCPTaRaLUQ6drJiF16V6doo/nH/W2/+MmnwIG1TO
IIBi76T02/ehhfWNtRM1dgETwZ90IK43MtBSCNMUAS5LCW+6ZLps5IYMHmRsmJ0sR8OHt2Iqpw4k
wlC9fL8s6/tFqrvO9rdMmBKSOTIjEPAQiv/XVEEfrWRUnfitYM1JRTPFdq+wkqXihLXzyC8pzLu5
TVPkKjr5HLyTunJNQBgO0Mimc3ZdFx4Jlw4w1PzuuKLZkFEWIVRNRaU34BaRjfDfThDjGZuBeUSF
suDHts+3gtyhrSxz5NdEYR/HyS7iJa6mHRaZ5DYGkPpj42H/F/+RcOCS3Y/kM90/NqGaM4yiYghO
VJCtk9SHiyX1zJU0XLTbIKpLYORWXbkeABE4HwP5o/WBqbCGWJPwsAMBmS4qO9//j/Le0dkKcaM1
rXSGmSkJkdg/8eIb5Vi+CzUTij57qlvEmQjH9UAb0P303X6XhEoAg3Yn2Y6jdKy+2gftoWCTtO4X
mBpYt0H6ZkW7uk+2xGiOzNSiQi4J3tPFaDGj7/OICEJEHYD22LRqBSJOHGnkiZZBiY2VpyQAzknX
VrfxxCxp1NqpOhi56I+aiWmA50b2DEoeKUsdEfjJz/9SBTFaEAIBr7vSn6OSK6pTP+u08Yk+TmCy
xUjwlgv8PwXdmJ/nG36CkdjuRre/5u1NpYT5aOXaAfBEBihBYiZUBbr5OKMNQYsUnYNbna8xu3Wo
Zyp/jxq4DcZZyj5yjy74xd9YAW0CjyThz/lPl69rr//omOAh1iXaKnvZdKLU7diu3i0BVQfrBihF
baF+BFznXjZQQJXNWSFdiVYMLXXiwuSTV7xZW2xjirhb3AZ/gigxAFGSmaGQBEfnomp8R4OCUmcJ
t/xJEKfVWEgStjrYlly7hv2t3NVaqppw8Y89FP+hvSqwwrBuKGVVvbVIXDHgJmCFYx/iVX/rAA0U
6H9TG0p5Ar88D9NG5qusEfqL29ZZ9Lm/1YPsgpbO1Wf+3dAuK9T+ZGiaqKe56IEtn0tPkGt+iqKx
NJtTStJaKEn5A+NzyrxjWKzdp+g774f9I5HyvDCsRHwnxRqPaDFURWh7Gi6eCtHiKtzK/RZf3i0l
wra5hCSASrZWgQnz8cTbIKlbKvce6o8eh9OyTHX7Fb5A5o8jWNpwLLD69IBiGbC9B/kCzAA2bFcY
6NKgLFeRIPbH7xGqmXgylK/lUeCkG9Ya/6jIXX70bEnl6qqBL3FNwhvUY2ljCeQQQSwy++O1mi33
irOjc6/YLTmrWjaUcWPGiNe3fmZzrYtqt48f3eR0P8KZYdfBmDPYGZvU9iuSgUoudb3kN4wbXnr3
/5yLvvvqsDrYflQf3HxgWAPb9gB30HKjLMfQ3Unim+xaoDjq4DAnjAIegpVWfpDaD8O00zgVos/D
As79BSFPGkfmpmLRMQL8/g+0c7OHXh0cdXcNM0+tSZEUMt1Di6p8cH21c4iOK3SRrMUcm3kNkE1A
i9/MD9FYj3ih8wPLx3ocbyljguusXiibxaI9AsG2k05DkxksXu4bSxlecU5Y8GMYL1dayyGX6ADS
53/EYLojNZ4RZfcNb4ciJpu/P3o5O2LXALbt3wp4kd8V6rumjoJlYafh1qBfH1utN4HZWB48ldTy
uZG4YC4s+Ee35q6gi51n8s8g/cDoS2ND6VbP6/UYZFISYAByOdVkBjxoZ+37aWD3fe7QfNgYyn/y
lbzV1Ons3qntNLUVqsWX65H9hkUTVPNv3j96/6fvb+EVlX0AAAYEAZ5BeQ//AAFHUrKmTp3LJiFJ
Lmsi7wuMbw0een2OAB1qnOP9InBppcEv4PwNomJ0mE73qMA1NVZ0heIT/Jx92QipDjFONt0vMf+u
Pzhz9Ky4IthWXTaZBGUxaMkqOo6lKWnWBzfi1UWvipYTu7wEl5mZEJjW8BJxtBHCWGXZlgaxjGyq
IyFPOHAugfhHj6M2yRXOOx3XchS36GLyPbhEC6BJAlUuKMLrzCu+1Vu1Ik8jg7igu/ghuVZMnT5b
T7dQatbRqBUG//7Ss0WZMCyauMCMc4EV1bKVLpNoMFlMKqF+CZ3i8wb1ZfMP+Gilydbp6JX25PF7
ezxX3t8FUYJH4gWnDUYbjWrX1zB542AEAGMT2arUzhviPMffQqSCN/wixnxafBkLvWCnuXXjs/zz
aLWZeMfaESobIAQ1nIIdRtNbnt+44m6ZRzjw5roSFXhF2nBlTyRizevj5FLcjDsL6IF1UIyo+EaU
FnRlcnH/v5xWR6ZIMEr/GYNqvLtpkup0RwRLty1ll3ZyzX3AXxDAhULY1aHNm9qIS9eMiM6q0kEe
L6PhPAKvH0vrAPiWT8mMW9SA1zkLdcBQJiOu6U9yd3fBGmcNnmpgW9y48FHuK6qZ4tSs1uxbxIXJ
06gYiyKcPnBqzAq0rdjdZwGsNsRKLmJAUqlRhu4MAy0ibpprLTgaYQv9WmlFnt3G0+a2lZc3psxw
HzYc6I+gpExLMMUwlVjokWeinpiPLsrlJq3VPT9ZlAACbZOYcfPixzN6+DPXlHETOgRH/vRL1Vxa
4+0A2T5R6dKwCKI5MOm1TLxuzvYcOkIf2NIQTRAu2fjBQuS3h2Q0NwrbVZhKMGOFyxv4pyEiXPlY
JlWbuzv404CTHNmWVi7GzNO25xr2aoLWnB9GBDNvzyTsznu0WR1F0VyadzjkIyCntSrnSK05Oq1l
12BqkadEKDVNJCbb/ME8QEl9iLoIhQ6L3Xf2LpVN6VOhXGd/FoMXbPdTHmGqp9iJGUE0xLvDSkVg
bwUV6EA+R12YXKGSKH9PtozZpgYtJmNG0UjReKzDWtKwjPHfTXJnasDRDU7RDA748/pEpjfGjqIX
PPa08ENchiL8hPI7f+jyZq59Kq3GgWXUtQUe3DI5Yi0VGGuz3GfvMO6+t/FtAyLao5+mxh/W4B6T
GakuUVXQ5hlTmdf4QGncQl+gvCZTViKV/0Otq5d4Z7g8NmZ1ppSVtW33SRpdvr6ad9pAZKbXlD+R
edYNZYF9oB05RCu9TpRJm1zOY2DSru1PrJDhCElZ+cQzab2z16+x1eJ0oXXuu3IYKmGouriVZmhN
qJ+P4bQ/b58SganqkhM0kHN1ILAvmtQc3+5zQLMtddrYBtRvc1zSkgI2dn9TEaCuzJQwaHTxBCM2
y2pwcCILbcqwoZcUSavk6HhNp0WnulCXUgaCypyN7MzCMS6F5YGQ5kHIXEPTZYCc1qB3qC/8X9pu
i1TsxJOhHlv5DHOfe4LjV7yv/e9spbUVKEKfrfu4fbX4RSo26e1fufWtuRspIajF4L4AbjxRRz6j
Ry10LsyqcA2bYi/2bWtLygrW/4z6WtuhI4zb9tDZXYK48pNvyJwKqhz65PZCJmJfT+TnGMdLVapV
TQ1Tw0ctb/W0Z2zf4MNl11NttfJu4RaYEuC0ny84Wcu8ARALQdbGF2ev545Lwdin/+S+PCKpwtN7
F0eVUIIx3cZs+yv3jIwBfYuUA0LfcK+L0Mt5QH/V/g4XkxY2n2FLXyoWZzzbfVdIjajqN+8GrFOZ
a72/32oUamBrONujv/GVsYvj3OlJ2VAq4dAMXStmDsb5DU+x7JzrzaT50Ovx+Pb8Ygm+2KEwkB5w
PDlAniXKyocO5Ga3EnZkdB+QLDqhCrHLzQkBTdSkWd3XgF6yNrbc4q5qunWivQMXYKIbJdz5pPxy
/5t6mT1uYSlaswEA1y48ubV5BcdPBh9Gbp6Z7pWz1vhyn/9MXofSfNAE3IgutbEmESMK7tt16ix9
XKVZ8ZlP0yj3IqahvYqPwmAy60QO2OQg2vFTB6H10gh7dlM7MWhkQQAAD+NBmkU8IZMphBP//rUq
gAB1PlKO9dRdDvCVlXZDODLlUgll/XsoG/0Plb4GZOzXHZlZv3ZhyQFdiuY5aTSW1+K3RRDx9Bc/
rN6xQ4nwy19RwmG+z6rgyx/VVhaR1RNYMtqpSvdiJwyl6BUiLr6NOupK1F7+dtng6jPZqTR7+DUr
d8INSQ3G9umGfDcQX/gNH/7g2HccvPcvko6pSC32uglRI6JH31CjO8au3YsMKPPVzXncbVy867vl
DhLWyensgdsLSZQTeHh0zKAH48DoOGQXhy6/svRdnHwV8IVb/tEg4/yuM0sAgPAEBrKYlwkrDhGV
CPQFwlK4IPCNS59AxVDlHRNbKbMBK++1r1RR8B52degvp5sPukW/vq11LB0xvVhoXL/fAPwG2fNy
oFCyadbwn8N8+cT1SSLiIiU25xguLDYZVFs5w8rm1cZ+cYDt3lDf4hoou//YPzHrQz85Ay2Twnm0
bTKiS/oCzSQp24tRrmwmUSE+GNxgEsvNUMKoR+N71rb4w8+y4Ve5cPFJz7fXcAWFyeXwls1gQTxV
4h0GRIVOWpqvMBuSmjqju/9DtkPGWPgxbD5KBW8muj64Ao8oKVS2n8i856yaBoSZ5IqfHu5NCAI5
BcDhaDCpD6+ys5RJZ/U+gu1lftQUxmM0NHsCoSpZCnSkXrbqMgPXkCuOUdEB0k07DfUtNSUypo0T
rnJYkFomkCAACn/ajvDP+HWoZxrO68b2PTuCDE1OV3Rnjtm1O4qgpfaa8e2iBnySNwnOpoISvnDL
1vspOKcSxV3grMtvjZCQbfhaq5EpxaUuE3mfslR3agaJSB+qsc5T7pLOL8DKOSRiCmQl/4JLkgV/
kSm5rkqb72ccnsNtffUpqtt5nfI7HSSbKhTiE2pBCU2lCImsfVrOY5xv2j2GV/QFoWUAbuRhlI+c
t/urrLLQZhELLSYgWZqKyBCV9Sy6L9QLkv71yCiw5pI4Z6qvx5olMWKcnjx/jnoqTANNf1dsvu6q
+wa3ATk6fNUe1Rf4hfVC5tAJWdPR/inPTOCIQwlDBly+Tp9DnhgtPzCNE+vGz1Rerca/CQMVotNV
VddhPMjOHldbVmM1HO7Xus71NX1yguBvuKCgNY2BdM/CIlYmoh44TeMVG6WBPKnsuCl+8pobNJmF
nVvl1x2jbBMs10O0r96dQJbiFgCIgA04RfcldfvT5E0AFcF5nE94JEwCl8sTyDDJ/ssr8tQXBNwx
z0CIRVAltKmVc4UwXOZFkYzxF0KTdeDqqjTybRR80ir+R8WNkJxW1CuHw58QwHm/1pA/wmtX3N/r
0OJt0c6m5xmdaueWdEszXZ9RAcB6wpmL76QQNhVz1TtIdtccH+9TZN/4uUKj60LiGPO+mrphI+qD
8ndtwaXtZjAUuOrUrqEczeXKNT8b3nq3qRsLkSyU5VIJo9+rbOWCHU8QhpfkIds32p/xqPp3dSgs
z8igAOypgPj31x5ILEYch1I71nZJmD3A3w9Sd+qRnfjZkIKja1nBV/LzaqTac7RZvDZQ0TKfxlVe
6yHBG5bHIEFNadshpWVMufqoh/FgnFyFi4jFwYg+y8GjnfksKCbxsnanykMaEasldPOjBbHsOMuN
MzDilHn1gZI11O4PeyI8pczg5ndm67tHWRX0Ho4V67QCdb1Kwsuehwbgz7qLIg1JKipDdLOwkVAe
hsQ6tbQkkhyMNe0mKrTsS5RNfYlbPBruc4/CeqadfWVa+u+em3ONlGNdm+OrFrwHabeqjg7onjzo
lD8Hw0ZuQevV7QvdEdLDM5ZWDKDwhf6khSZfjTmLsO0oA2A7l7gDQOj+48Y9a97cr+ylbpR8cQKc
64lw4ufpFQ+WvNXfAfBq05I9QS7377mWONMpdB4AProz1Z5rpDYx/5OMUjLyY1BHQOfEXr8K/QHW
TXtfymbfKlpL58ApXD9zELaDU62R5yL6cDx1oIzcaF57JovprDFTxolx2U8eN7jN4xUb5Jzn2i6L
U6yL3gBXS/V0eSDFd28RNWXLwC7/VMD89PSz69l+60sg7THhtalDlfVO1DvQUZWSU0KoL3g5nLLO
CXX7k2KhDV7TKmIuVQKb3cwXSQi2NehCMuPtgfR4oEiKOupGwsqH4uoGgDtyk6cSahpJQnyqW4qT
yUPFNvdtVjmDVWofKQIK4GVoENYr3D3DkO7MF/xSp8B8VPhpD31qSVhvsNc+Xdv8ntPaYXxdwKd+
P0SP6d2CkSERto1uk6nmNujQXqMoAF1Aqj1WVPkh/KGJC61v+X1VY/dOClJJ5AhjfnxVydpowOxW
1eYmvi6ePU8ZI/9HkZRLDIS8gPX7JmGA/UdtfJ9a4T2ZnYNpEwfpxV1zePvfDIML22gqAp9u1TGp
IWug23FZ+DwZjM52ne0cnlTZpzjCP2GGawLOIf20tND9UwEC2Vb0tAWiO3CUOd3RZXFuw92mv38p
oB0OdfeucqO6/2SjxzNcE6O1h4plUPFqbHyFARfs0hyF84LWo+dcbu0F0KkRGHipXOUS9cnoLBaW
19tc0hXJ7FlLhATaBzjHVy5b5ZBWxh5nUarcCqn08k8+5uXmIsdZ4xcS/lEidD9VaPs2pvtLAltd
rvSj4PsSz2BUUVk0mx2WRfPgbfHbAMdGd96V0bYyh+zdGVAVrvYz74/rQqzVKQvXPcJCrNALd2HT
yXBKhdRbSunbprt9SO36I3JMaFWfCdH8h/IBI7zjb/lETf1NquGVDZRQmkyjTmZ0+EGZcZ0gwmjN
jY9xUrbBcqaT8M7JUdzQWWCVSJPeA3SjxqDdLacC/6t+X2G3+r5BZGz4iMl+EQsU6QR07AXQLM0F
nzkFT3NvmIn/ct1ZiiGn6ttpH3v65j4CxgFPUtrN5jLCALSs25Y0dsl9WVIcMOdP7uNMvP+iWXEP
uRlPmSXK2DH+iMuIkGNNrDP6hgUio9YBywuvJoQSmsiG0U1O19Mr4SsZid09lNhw0obpApBik5Xk
/7hMFD/3YiCUtNGbMHo0dgLG79YvZiYdjb0Zuz/U9KXCLGe6LxV3LgDLNSypvQrVnkT0IWPR7AY9
3bKBgJ/6lf9MdGagD+/s5U+oN10Y/mkaPyR02Jup7yFq70RcivzaievD/IVfniFqPBNMn0DKVjZy
HuKuQtXYvn9vXp4aCKvfN166zEt9w2duoU+NcZpWVn441nky0U1/XXJXbB5ff3TQ0bfrYOKcK4St
Dhbln+wEccM35eXNSo3/iZZXTEFvZ64NVXduJtJaeu5vPB/bG/RK0LC6fLSleKPUnbB7cLkU93mo
7zGvLOeUOvxOI7jX9fus630v8LRApTMDirUr7ZILHPEux6YslMnFJczQ0vPVVaqefIcZ/vApMLq8
RL8WzpK+gqJhvyK2zDZqShbwYcuhcdETgKqsEXK4v1dBMs17WJJ1WdPKT1LryGgGeVSCzyV82oUX
ezu3nz5YWLTG7OTLQJweEUwIwAuU2cuLjIonJIlcB1p/kAFqhjL0NPKbSQb0gYqP7RgyRdrgDUpT
5+AzlgsE0YtFXOCWvDS4uO8f+cIFspTQzMHBuBDJOeodlHo6w3nSA+ESzCRLvJ7SelRkY4beZToA
G+RIVXPIaGuIH0P7nmqGYxjTnIDRwjEtgNStfw1fHGquroIf63ExcH3W437ub9Fd1qRQ588nbp8U
QS+8Mof9yIyIs/uvSaJWR/opFhTG3ZTlXlAGFEB/2xw4/xQvu5jJ03IePrDHSQrsQ6iFI4GJFavM
2+GM2reTezNB8iNeqDdZAGlikYSxNOconP86j/sGuHwlRMvsN0izr0sedGZNyJ470oGGOgz/4XTl
tDvKtGz/Z7iJ3Q3neHqIDbqW5gvwwNUUth8vJvziLniY1bb/nJqS8V90eLfmLx8M6ETJs1t8ij46
H6cnZjLjq/veC2CFlXQHrTZHoRauHVPpNgModhu1xQijGZ0AK7XIw2w6GOIsrYT6U5Zyg0dC6vi2
nNfGXHPKwaZIJ/vUEq2PGNd4tQpdl/8wpCA0Thhym8b6FIh+G6Xk9CMcvlYxnKA9N3Y/OVoCKtlz
xt/JHH9ocavMJmohm4Z57xdHTiSIn2oILN4tncjIbB/FNgdCyWpeMqAAW5HFVSBona7WbQ0M9zAy
r+uRdCHDXFMmpJ2jqLNKANYQZH/BvgwtoFB0GZjN3tIdTk5B8e35HujROadAOEhvBF7n4Hd3GYdY
5r8oqpc4Hbvgf/YnFy355i0SN9Aej7k98+0kiHNRAgDi+UcWPMp5a2Qf6x3sD4hMs3YrVGo+eN3l
BYU9Exg/zqzytjg0LLrvdMV1hrUJ24Gkd5+77NznntKH8+/ZJyxKgQvv/oznxF6RGKSWA1kbmlge
O6Z0vEbLSj0adbMaH58tG8z8uGTbad+YxfazfkwsZ38rdOvBywaxnoR7cj98EAaCUktPjsS6FoO0
Oh55GdbmvjPsmcZ2XjzLQLOzvIl6nTOIiSAanKUcCGYH76X123MsCm0s1B/tf5GSCqrk3wlIjVz1
W0s3+z3SltxuAV0nIswiYSXSpNo9lj2FEnjvO+j10al0ZdEpNMJWaXaZPHy9Ryns0vGNMh8ixFRM
TbT/7M+c5X8D2PJYEeGmCeXAvjKd5081YswuRBDyDpctufSwTr9IlN93c7mOE8TDohgORdLyb7hO
/fqDwYIDsKomwUjGbapElbVsERqcSKsXahwmrjnLbPUVdaAW2pnJ+ftjX5iWyRKPmFZp6vQ+aTs4
XvYrbFp/CMNcleZ53IolLMbXBoK6S9czx8e4CC8miP4Po+Jt+HcUK6ffUESEexwCyEv3vLtC9vFk
oLKm86JES3w3wsASVJokUIuzx/spf8NAR6PiomeRFMzsSDgApXd985SPNMibb4tiPJRaoyFtjKrr
RQrF1a0y64B3h2WcunXUfGurCc2WtmwS3XEe9lmkaYNp2JLNfq0D5OOyfASWQforqaG9D8ja6N8g
LfiefNnlA5Uu8bftn0xi6STDDi/RiLwVqtZHmHIA1gVjcKzrrKrY0T4dl6OFzQJgyuLoYyyo3ywo
SpkZsWsBvRnqaGDETmObDixbco41vfkCqffy/sjH+TMtHPhWcucjQoL4VRjTf/EcpwBt4SRD9ql0
lsCrBiCQsvo2asl7bPv6Fjs7VD66xq2N/2hwnjhaLla1elgB5t6ZiSkoePlm1hvPcPiOnGo59kzs
X+xIJVs078sccQWJSILqV5LsuuIjExichzzAYNl0veRtFZ2U4IAkedTwfJRsAo2qEtrefdJAMB+u
lh6BohrktspeNOb7ojiGvftiXKU3WyV0R3irsypYtUQm3cdDLkB0g+lWxdo2a8QY42Ko51P4yDnk
CVCGSihhpcDQcs0WcmJFaI8g4EfQa7L5MgoFvu+H1N+26AZy21Hv5spELBtZxoYs8k0AmOy7Wvf2
xup1EZZlQAAABFlBnmNqU8EPAACSod1D53fqlMAITtlCeaTTYFLOgNsQ4x+gIpvMzqizRT9kjF0V
OUw9yxAdzswYmfJNeGnAAArTPwsUfJLiOXXealecqDbpwh3iyok5krJ/wuZDDLgdhKZS9vUo2/4n
x37Z5aXGiWyUQmkmQCCP029+FN6zAXEJGbfYvSYR3TekLLmtcDqWtQOihVVyhdhGHa34QwL1+54O
1tAXA1z1iqaBkaVIv6ftTQQxYgeQhH2vWzALw8XRupjq4xR9wCGSOFCxM0WMFYBTd1HlxttP9fgZ
IGAUyFwqOhFctS7eug2DjJpXpcGhCneiT8VaWSeM/3ryLsyU86dM5mFKcH4gm6eGDdFZh8j3UoO/
nT0hut3svukXXhA4oiC/08kZy9eVmA32Zwi9Absd2fzWxw5vdJR0l6RVyICCuc8Xv98KlLNfPve5
Mzoh7Q5hqRrPv7T2hmztGEq7MSbaW7HFsmws7vX5dCEEi8ED8ovFRgushZXHf9Rj5sxHwU2H8CCp
V6zj5mX0Rab36X3e32+uPyG9fwOUKA99tD4ceBBIEwn+tUK+X9Tf5Z3ok11eY+Gs/3bIq084Po00
g4rJZCmtBTMxQWrgpCQ+rKvIZchJ4hTO2mYSdkorF77vnvCoCm5C7qKPh8C0pdqoWxAfXdqlv2J8
xsSr9/PtiHl/+qjFssezqptWBrNtgoRX/Q8/QpF9bOCLavAl2VBMpcsdWn+52syM3KYnPtydBTPp
Ncyp4eZIKOIsWr77AalkW/EkGOZPh0xf8lWuQv3sWOZjgtzP2UhTrFKvcLr+Bzt9PjJxrsoC1tiJ
2IEPTmr17U4EP80bop50BieU93oZD5a0UaKGUEixkNSD7TCTrjwnufZVxN6y5m3Hmfd0sFNX/TLC
7tgM7ne8Qt8uaxp+PeVKDbkO7ktZb2ocfAm14xzZ7w8T6ZWIAcQE5eMntNTjP2P3u89TYbPaBsmH
xWHU2Lofsqxn2SJO37yryoOlRKbmaG7ADq7UhpRyOdm2C8JbC8IY2edX8jUFHfl2S6Ol2hFnj+Gi
9W7as5sACV4E1mgKyYBgb25aKE5cCWpqamniP6bYKpxQUNh22H63vZ1FRvaNeZEUge13VG/x+45k
wp0gwkGTZ0wymOBJiXFvkAOT/jAI4Lz14v4m6ez8wOvbgQdACBqTBX3R1+CV9SrqBcaL+n8796GX
A2t81FDRdlA1lAFTujzCfeH7tM5ngcknrSGTQbhIzRBXaAlOHClHtpmPqzYh2yTIJ6j9myBuxqaw
aDcsbpGCd9K1lnN2ji92DoQu8ma6Y4SYruoHTcA2GhLbrYyTzJCJariPOIIW9+kuZX88k4MAQ6tk
Fb4rOtKkyKctI+suBjvjAuwWn/XWk3UtXuuPCTol3TUIO6e0n+mnihk5hQOkdIiavKrmmZA1Dc+v
s+8vvgE4Km5t1kkdeKl+/BKSQ4v9zSJ4fFQ5uWZJJN6DX4xlZs1/6kEAAAMjAZ6EakP/AAS1o/Dq
k7chMy0AGtoUz/7DjxUUCiUqqgYmzrilvF2H0iFdA5tTawXXrAUdm38OLRIXtN4mh+XvZx6AGj3Z
kXPf/0laABtJGlj2MZTXnJYLm8Thc6H087osDMLbylUPfemAJ68LBiCDrErQ4Titbql8axk8PNWz
//caj0zTEWp+GwjTnYtQWL8+Z5xCtCxL7iCXx9JL6UEPwzyM4QL3IpBPojRiMI4h9Eqa3cWcfmbW
jIlbmCmEQwUfoQgLJUhzo7G/ESH/ab6u65n1xVW1TDa4/KTC0RiiA0TSME8Kx01iwNa+OWtqduGb
wNrBijwy+sUx38551l66YJ84upgrutvdKTFZX+SJCu/VJZYkMOVigT4JEwsnfVkfMZEwiUCrds+9
+XjVepmEK22rlchhUizoa20ObLHED9H6bV/x+9fHHJe9SO7BS/t6CHjC/8eUYcHsrfbRn/7dth0F
0T6Gpc260Z1KkUbSRHnU7AKEozcrgqyMkQAZne9URZyQUmNn7aEGA2aLrMfOF4lbLa/WpwQexzY0
Lb+DQ1l2sxsiDMx2+Ob7MfKcyJsdIOQSvZfUb8MeHLGPEq2rZIZhxxIx2kYoxwBTqRvrJHEwVMPo
nLo1/xJmWOUlC+kSwc3LdcUXnulkWYFnkKUW0K+VogtIUYcwsYmhEmJDVenmYQvxoQGHnvtlnMjw
5x14xHwiGPcdSJwYCTt7tHnWm9444PAgG4ltdY/CPPFQP78qU27r14KOCnapnpyok048Zu/Z8kxM
gBnx+syVVAVBTEIQhdc0P3DIqlDt3Ibo61iGAHsjejo0z41eM0lEOEm7HuCjLyfguGlS+9bJ1I5+
AU+YV4wwsOTsSJHwrdj2+BzKO2GmiIuF/wQMWscj+eBTMkTRs4clOuOMOFHntOmgIn5G0jnmA49k
H9DjPjplMUReJ/n0AmmQMIqZiJdXCN2b2d4/bEqLomhxTWY5/aUt/4c/5cQqpZICtnnU0ia4BDow
XvgUNWKf3YthhnODQAGlz7dAKgi8LFC+yy74mJXoX3CFwpPO8WFTZyFLL1qR3TYZYysAAAZiQZqH
SahBaJlMFPBP//61KoAAHHjk5hX65JsrURO4QqU0fOvaWNBJkKe+b87ZZ/gssUAOPvOxOVNebZMz
GQysPfnTPljIs3TZsrx5/cgEYDKLtEMdvfx81l+7Z9ZASFWad8B9vL+VneAvVwowWyRCCV7JqXcm
M1NtSpopV/1G7nsIbJhtkf2SVJa8UQOMV6YgGIEhR/mOPNGrNDbQWcR4/RyshIY03838Ow/1OHYU
y+8UZKHMhO9/8LXTcAPtwFhx4dPRibaS/ZMvKKy0HAydA4TcEAsaarRhiPIDp4VbFV/rk6DXd422
0CNXl+0UMxCU1+raBL3I9h9aMeJx+rmnd2c1zNUugXgTNJU+1DGxhHPiWRkmSounVjpnLPVvWsJ8
DFIFjKwns0driphUXNFCM+xnz9Ea1HdwZ89WopTqMGGkjC5RP+WbmPScrk82mi/0XD6ruKQKIBh+
kOVcR4a0Q/fdvoITt6Gj7GYpf2GRTbzlT7p6aMqoIGmiXCI8Df6JamnBJjm2OlQSrEUhYMCkSGzt
f8wcc7S7zZCiwdxm79exFqaX0oCdkPAYnVpa1XAgBviviN8X1M61TK+2Yr4+jZ1Slh9SCEQLCli2
Lh3Kvwi25VezTJYieYYg+csLC1LW07Su+g2pWEoLBDO/JLwKN3YWncu4tiP6LLnMtdcZJWBXd3l1
GHeIxmTr3nbWqCniO6TagtKFhGHMIij7Zv2Wx6QiGooK9XQA1y4Poy5LwBH1Ojorz87s8LmmYlAh
dkpzwvRst+u6Neahih8ojmttuJk1wpliGVd9w9RefBHlYMobF406KeEDwqDtVodudIDJGGu6SKE1
8KMSHehwaRGve/dFAPOxs3U4+Vr72a7WyvkZfkWjQqt+B9EX+fbEK0DKQTcnMPpA7/8FYwgfsSKM
/6UxxMlmEyS8bAEqU3IpUF4FTFicAaoLKnONya7iTaXEN4LMT+LOqc/lOJJPnj5skZDxu9JntvEL
fTjdD+70vvXbXsRVMy1Bngffc289/9zjP/SJf9Cr7VLbh1tZwIirSAPbClE79f4/LHM8LQ74X7G9
5wWj4jIzFq3E3N4kfNSAuLZz5zkxvX2boY1l8EWf+e7aTecXRdp9huTiWLD6lfAdnd9ltKBKSk04
SlPgEaB/Bs8T6NwJL7AoxxjCW2hd87tSolYr5FDyjG0epIKVjYt1qq+Uj5oQKIePhnHrRwhphmYQ
WmB4BQi3hM53Bh7nw2oFsixWCE1vCHQdbOx2WcXwAMtIlVPEnghPzF54kmYAnge9qukAZigKk1wl
OEWTjF+8Fi5qv7SWzDPG6SNnK3+/cBLar4uYH7SVDl0jUaqogK5kEV0a2VmvVNdkrqr+CJKxH73b
xFSrnGmz2fepRiUDcebnfEkpUGsl8J7m/4A0UGqy1uf/4IMH7sPiWWNlGLXLs9ItomzzMkRVpn+o
Y5xyt7fWmnQcMwqkFW4t5WCZjmypXJddkt4a0dyOJ8sLlQrNT/5r2nqfFXbcdD35hsharro4bP5m
gGASWGJB0+nWcwwWtXWTWX+o2skOMy9uZZKYBFCUjK+8KlQ02TYpqgQcLiONYqo82RMaao5F3cb0
ASs0s0NIZv2lDdv/bLcfwSHPyZwQcSN0R5XsypjZZNBL4nRDCkTriCBizc/T1YjVkCwuXcszOFIr
y3g9jKUGX3dS/47QVLpjXCb6wf8xMTStvH2dHVPxnyJDjNMyyuvTUaTuTxrsAGDwSCt5RNbCzlmy
Z95IucO1eEOCAaVoAkn6/9rYfmebB17033PLHMO4XBf/ozqsdFyINC2JhKdXGwsYrwWMThAQrC27
3dBYlxSYPsiFb1un+5Hg6P0s+LaWSNwfrP1x3GqBSNco6+Bc769Ov9PC0CBHpmSu7CGWbEi/3say
cuHMZQ39EwF1pS6p9IUFGXPa2EJSmBSdJAtYYEEGvZKWE9vlflewbiBi8Cd6APXHuxpkUs0YpY4N
ISzWWaAYEQh//qk7ZbkRrksWXcZ27Qh64+uvJWprlYsZtZcByvLgrP+LdlB5sjZ0Do1JPHm79AvX
W/edsfBDz08NlYEselQRVOrpkejeTVlnTH2QJHWXJVJpgn6QYszAdquU1bw9t1bJghk/6p/m7qtw
haARXgezWbo/v/59Ujv6mmFVuDI3rwVb2hpXGjdTkvr1v2cAAAH9AZ6makP/AABPkfNQE7fI2SI2
7LMbfNG8PX8TLIlmbi1o4twnMF+nZdMBxlViqetHt0EXJAFtm4O6W1TI61FGhE/uPvkODHllFRNy
oOsgFMsDsz/0qP5j0ZDgyqFvOEP3sg+2HvwgwiTH2pQAdCSt5EF4q9pa63c7jm/grQXLA0zaub3+
1gAAFn//v8FYl2XWaUdq0NjBqU2HB8o+p5HfshUQPorB1A4EecXp8+9pn98O2eOPfJg/DBBLQoo0
jjreXitmU4U6vgzTEPcqKgn/vU0H8BlDhInyXLTn6oX7Olw11Y4+RQpCK6BOLzYcgLSQOLtQnJQ9
jqCPoxJShsJ9nU+6hn+Gh8HqnVSIpyk5rVFQ219/QTFZ6C+4MzZYZC757ZBOsF+yUBPZ4ittOI8T
wYGZL8E842x4Onko1x5oDxm71YAt1sRRIpWcZJNytE5J8sTOP9UK2nudW95zk3Lwl1qCANxI5uZ2
XaQATsW8LUrNKAFld93XgdIe0btRdO5Nr+agILSy/yHAAPNKvABCuH+BXm1A5Qw5gDz5sV09aE6U
4IUR4gG828a7lLxMPd9C9ZPUBmzYTO5jiSaXnz86Kqq+72TycfP0mH1KQS8V1p7izQ9AXvE4wQOC
OCEJNmmCOOKgcxU+VqNpJf0WzuyjBEcIGr8IUnXqptiK1lEAAA5HQZqqSeEKUmUwIJ///rUqgABy
MtG8KyriR/g9o9GLIkyMxu7ok6gA1J5Nr/n8EdfTyYj/Blb998v/Rj9301j83Af//3V/lfboJwJN
gMejDHqs5/okCwzJJtz7y0m+iOwzD2zcK6Mi4BnKNVp++y5Ts2ucX+RmF1y/E1WEpyxGDumN1SU3
Our8o/mHH46/dbG9Ea7NKNhz4H3Vtz3u84mA1r/vnFIWKSiD4wyE9f+1sMjxAQsyuA7cx3IUqoV4
67G+aBHG5Ms0bt2/pMirulRmkOdyKyB1oy+l9BzhBdTbH90ksKZDQQenN4BNI7WrBkdCkmdZhMwr
3LlTF4/DwriV2QHX1rpNmGb5LGI9TO3qOWDZUau966PU19NyZpiftvUE2vzQ37gIPCtZ+DP9x/Xb
OOFMT4xT6o//6KEhV0AYU0eNn8y1qzD9EmKbAx7jFRjwfmlFvkjIUjZYVCI5luXsKhlxLcWm15yN
EBnyZ4kEKNAHNtXUURikDMFxr6eiN/VFd9NaAIm6IR/UPmqneHEDv82NoVyhW/8CCUXB/ksk36ds
2aJFUnH0ef9GsSN8WXt80iAuuv6nx92RbsJ6B/MStzqYfUm3b9GEWcB87UrnW6EPie+9I5OrHv73
58UhbIUdVXW9tEmyt3KvV9a9ufnicg04Ya6Q9ztt/z9SvY7TYTMwnVBeAz8Mz4HPCjUYM1Fvp38Y
kBDcP+9wXY3sPuuZO1pflwaPiny0tGDN8H86p+P7PigftjpG5SuD9OgsHyeiiw0rt0cpAQDmUaOl
cpVl5wMGYveNjLgUpUzwz0qnxJwpn+rbBZEwLmhsmX4mUv0qGf5k9rvW1U/caagDPw1HCcnUwJLO
dJ4ZMxEOGfwkjlECP6t6F/5HqFNrx+Xh2Cr8n9tuejkJG/Ug9qEgUTS6KLudNMSeSv72QevvvEU/
oYm03enDkd2NHNqLiOi7NoaYM9m3yoDx3DYb30vtBxPweZJyBNxmxKHHVZtBIAL27lEIINzK6mFq
8g5Z+68sZeseixIqzAPPRVC/vZmalD/GB2kk6l9DUAn8UH+uBPbtMHDCOjYbgm85VeLgYQJuyU9h
epGGsEn9QCc0WE5QSje9qP6meebKPAqVAOqzE0Dtu8WAHbrWoYhDV6jmY2njNNQp/aOwg8zAGQBT
me3hGH+17syRHUpB9qVkMnUrg25YfwDtSL79t8QnIh5NZ19CLAclrWEBnKdFB/3c/2jGz+5f5njA
bvB0toGB2F5nYGW4WIZqF/TJZBoWTfnW33v+rysVdhRD1ttA3AezECWjpTi82y0YbE+twnC8kL8J
XusnHKtnjC0eUaK0tqYFsL5q50kBa005qCU9sVga5goRkVfoCGjkgE8fGiPDYEwiVeaMMulUeGff
KN5a6Cl80OzkloXSnmGLLwTjwUScpuBk+YRYNRvqnPWzXLPbUZvYXAwdIR7ofa1vyMeWceuIQGDP
BFj4TJcNhGNl8W8uBOGKgvJIeip2SmsgJHDBSGs70xD95NRZ0nDKOb/fbM1c0jjlNdcU94NvVM77
Dih2IqkW/nW9KDnlkqh22yQGU39EzlAPjoU/uaa3WN6IXs+CfWSCV+T+FME/4mL9NYGv1bWPTZ7F
w63zBkhEpAgT9WsukPdtts0JpkfSaj3icWM4TdaNJnPM8ZYB6JRPOB5M6I2NwlD0qgIgn2ejHmrV
toDWQkWYcdXT+hcG/ZZBiSD+7PuML3f8+xVxwjfiMP/ECRaclzEltOemIHgys8YHkrQygPo4FZMq
ub+GhvBZysnJk5BFdQsDwusZW4dMkql0fQo93woavAzqeUAOltuH51CiQMQbdEXG/A2kWpnm9Nlg
LzfHaHz6P6SXLwCmPPxj4N733MDQ0ggkFJFYekz167Xy1C8LJyv23ZgfAcwMSSxf9WKzX8Es9U0O
fReB+OLoPj5nv1S93MgieYAXW1z2IlMYQQhsqAI7QRPrHu2ICRrnMF8QKzK6bauPlJie5nOYfiAg
R2x6S1BfIcy4lrCHKDO30J9m68y/DesUvVJ4DFvKD/4dgkSJ9YT25e0KNqnrjyuXycy9re3ehcyR
RZpuc5Nmtp9fG8ARgdzT2Onj/5jXkUqkO4UpEZihKQIY9kwscIGTJi/BXLSjHcUsa8fgh90MMelb
sVup37hCxlH0yv5PvZhD9gZZo5smB/WCbTGnr6iYwPg+t5MWgOpDFMVAEUWxdFShs7+iBFZRTAJN
c07ahxG/2NZDB3XdTUdvnW+TlB768HjnfOYzVY63BNZw8GMNvym+7emU+xoAm4sX2MJCbpwA620s
FLsJueU3ncD4W95HOUgCGQwvV+fWf/lDb37nx+qA9DSRm80dP7vMoypbbOMwz6ajFnHy2rJsylGX
Lt+wH+jybHhjtvJhMC79Z/Cd0dHCJS0l2vsW4sVxZjcqRwVcc+v2t0IsAccUpSHq5fDf9fch58ZW
d7K0cexdw2bRNRWOPxi7apkYhtIcay3evjjxNzh1VdHBZ+pi3RAEsO0T/NyLg/2jtZ0cRhvgu1cx
xek6zGkduNre4XncK40+VakjF5f2gIUXLJoJwDdJ3AFfqdJ85FwTrsfLIP54cJp4AvFACMOq6KFR
xevhn9WwWAgVZOtE3c6a7DHQL1n1cD2eLeIhcj4emR11dx03Sn/L6wtgs74luL2wS6CkZGFjHmhX
25SYDq4nSWLeekhZvG3CV4qpaBX6AkNSIgnLfH85paofolEOt5/lpNiEHAz1UMB+LW2I6i4rwE5L
UYq6Y/qNScaacfmRsgY6cNxf7ZUWmIKz57U41gzkTfuI8vEQz/Gx21MuJjYL37uEwEhzPCec2cM1
sD++ehMiMPzKQ7YrPEz5+emADE501Zyh0C61vcwqVhuZjfYemdnrIA8vVoVqhIAs8H8NSX3ur0TL
E1Zd/V72i/6bM80T+bxkRe/cBt8Fze8uNqrfN5mrVzzsqTC/B5wSOqn2IyfDuXL4oZhGvbEzmPzv
nzeGlxwxWY+3gd+OUZekX4B0JdIZXVmpFqbJKfKxsOJ600kQYtB0mqaFTTBA8+baH+AWK3sTrHwc
Be8nh1/zzYeX5SU7xyFZZ43eiX86M7s/PYTfskxJ55bQQ42Sb3xG8jPrsXMzaDE0dllfflPjiSS9
wwNTGpDLZIPrTGWMzmroGOQFxTpyE4xb2OHkf1FEzICV/hZxLq36lMxzcvk9pfASVV+X5e2/ZPB8
+jQHZcg+dPakEnXwLrCeqrr4Q9DkItqH7/yd4OosmlBkhm3iOs2VxARd7SLA04HlM5M1KXTv+kFh
qG8+t6IW5gTUCQcZWTRhjSL4/E29G21ws7l0UGQP2dP4Pt2kQ62EGLLokvRX2EWNR8ZZ27P99mvu
qpKk62BJbQlRXmwzahE1xivL3K13FmzSLgva3ZaHPPShGK/kmqNCXptvplgZH6dUMggWFyBHCckY
u1nMzgPgmGYU/lXFHhAZa2lJ+VNPb5GS2ro45BwfOtGea6rkO7/SYM1n6xm4+uCEMMb5lGUEXH7A
dzIG1crE4ZQL3EjWRaZ64blKo+6sntOIO2X+1dsnyJmvvA9bCloPOjCok6N052AWC79/JL3KkeuO
nC1B7Mh1z2A7TL6VmVkudfjRCuSGKdi2tJTiGSC0d88t6SPWMUAQ0Aty59ZMSiwBeU5V1jOJnyDr
VRdmWCcA7hPGiZdNHw7qXXVvxbB5tLDkdgIJ1sMjPSdfF7Iu9lj6WgL5QGmQCnl7X11CtKe/DTsO
zk6FXnFwNej3zv/992803DcKVS80zCqtIuLWOxb52bbcfXOJ/K0D6ExiIT4tXfyKV5Clxje+EwvI
x0XcS7nhj2kTZQslycHaK1y5K9WfTpPw0Y7b8YGU6OBUT361M9WgPp2JCPjtEycaKh15AoVeZ4Zo
97ThD29ef4C+0XBqOHwrIfbu0e6+aoOE7FlEUbQDZVnbnoC/KneFpH9ZCrBZgUMhZJNmkwMIln3C
xaNdQ3lkSnotnuMhIyaeMkIrsRQRu/Xh3+nc9SKI1Xsw/xxs/07X12NaNSUTgriNNCQSmmPs0Uk+
Ov08sJpzVnaic4/ib0MwY0SGYpghTQEj4rFtZne7h70IOeD20SgQyfarGki09k5M8Adpb2Thmlf6
BcF96Y39MKGsbHf4JKSEtzv9kztUrQVGRQik9hRUFWp2YWDEFV7SPF9BH1F5BhkBE1bizDzwVSFG
kJ2KlQHVyjo8Tcfh4HtOCYBhqm85ipw9SRjeITXG2gYY6OlT77WmQ4JQJXbFzaC5PoHis/TSN+Z0
LQ1UuedT0a5QLqRqOfdQDxJAQCESuC4nMhKIe+bVgfr9qEA0B68SyYXpbvsXtZ8No6WJ1jfa5wLw
X3WDOeo/FlYM9zpWGF58NmvKGXkGVdWDDvCredYigoYH0ciAl8L8VLBJdR2+MthaSk5u5nf45jKJ
vHnvPgIHaF6MchpE5ii1YFkVF8wWBx+6RhmPOR3HP7Tl2HfkNX9/Ffnwq5CsLfnBLYMHzNm7JKuC
DsOpQalHkhg0Gv2zTtP2Mrfmw/eO3W4zebsRax88w8Sc+EiBL4n/cwzuJnKTTMBd2MqytLOltmWX
pdZv5ZOK6vqa/euwJMk66zryRNUb33PhP/pgF8lZE17up+CETw4PeQo+nxJipiHj8ufm1aphy93l
zJo7zgBLwWBIvBsXrbyrx5fmMMnMFMubXSBjmG/ymx4yUjg2UIrK5IqfXCeYFxlVqOCqLrA9Qmba
H1F84MAXt2PeI30fs3LuYAPSysKc2grFImCh27f111EPxQRYcZyM7Xp/9cNr0fl5nJVUx9w6Qw5F
n6efNcCqG6GhIkCemRXtMF/9i5PQblmwa7BODpoH5FVr6Bs/paQM/AJl8khCIAAAA+FBnshFNEwQ
/wAAjqfHfeD2GLKBJjeYuU313yX0sS+1/SUhpNwutcTKxNBXcC670mcLbaQikOom2C8VTWVyCB+k
47uHLV2XLO4nx4Pbrc5capoANqj7v/5/Iw5ELEs2+pyhkFvAtxiRL0klMx5jhatDrDxCZ1tLX9Vo
0+DppcQvZZpAjnWG8kgcKtpR7+jScQh8jj1U9Ddwh91VFKGaMsoKWOpXLhEkWpTUipO4Uyj3sGP5
RswT8W34pbHkjIPnZG/zU/kJk3a+ZtHMZIaULmxuLGEptCtkYe0NfTtMn5EPyNNAzjNX4PeUexPf
Wl9ddHHbkQNvm1WEbntScOU21VSnxPnNRzBOXuNQKPmsDtNGa+Mga0HE6UQpAK0Tr9JnK8CnVk8n
cJPKyUFn1h/r/spLY+CKcPtI6T9yGGhMcO0Af+5bxASgtpEOcdOgDhrhOuuxNYt1kUu58xtJ8XI3
pCGdr3/ULnADTyBia4/kwTMxM7wg/DHlHonpCr7TrL+o5U/F1YiHzgYwybdPqpGOGt/itV3/aW1T
+40wyHxKCvz4bLjOpUj8SmATAH7SWzXAM+EH8AmKHZhsphfcoqYdNUlCAizqzQFyFVvXKjJSs/vX
6YvdAGfVQMJfm6IHl6RC8aDus+Co2B92GOZxSkFHX1IKdYOatrwIOwwAahjJigWjLbcEd8w3BAXG
tOjhv/Bm8s4Oz5zoe+lkgJYIvfoAgS5ULEqw+wVQozop5evMwN+Nb9j0ZOii4NivsxETPXp45wgt
EIqg8MODeulUGQjfb3y1KVSejms4Fjg9YhY2ARpMwuRcq6hFNoxQxcvVmL4Z3Oth3Pe0AInlA2Gw
dsOOv5KZwyrUmq0pgO9cKFWK/ueZ3OftPXC6guC+V1uwf7NUcs5ajNDTIH6bMD47EkDnm1mOyGfA
qwd1g6Xp/TwcImSzP64ja674N2X+deK57UEXp+MynerM72I6MwONVhqi1oY80Xp5On/jw6fz+HRc
1uKgj/HPqyabrw6xk1oDxCZ6vjAncxW6TtmQuiy+noN1H16kCBPWz2hx9e8afIdaD99xbJ7MLSb3
CAlDUCM/Xlga1Pb55lX16LUj4LExRdHV7ZfJF8NoG9RfVTINU3nyA6wqqVJzJ4J1T7ymWH1rQO8Y
8OJfTiUKJ+lyNFUKHYCv0nEfvzTYgyXnYma1f75XniA/CvAUdB/BaTwbq+dtNgmtyg826zHpKE+z
lTKBRul5jqH+I+Alsd8KD1nAPirLNZLgCfxPShhN45VncEhqBh8hUkTlhHnI2FLA+CSuiia2Az0P
qPWRgXRIfZXVOha8CHMJEbcAAAIPAZ7pakP/AAE+TyvIdtLMqUT+/u2HuM3Yk3oaOz1To/WZ7uA6
fbH60uu18AkAIQMrBkPzi1SUDRynb1VW++UnrNTYh0IL4aan+MwqKmjVXFcyCQEnOgC1IPIM4zBQ
3nLSMba3VS87uxH4rQ5KoCexpxQZQycyNgODcZGnJIQEtVa3Dn/5RUg0JFdGpLLTbAfJ5bl4f/TA
wSZ5uF1vTHGZUjt6Qdav5q5N+RffLeoy0Kli34t2bQ5PDG6l81kKv2ner6urH7cwc1ZZG8czXgQb
5PfudHqoBEnlPQNmpCZG56zjJU/jYuQoWW5ax4vwGAn83YLkExJSNN62VIAkKQd+4uklh3TyH9QC
Sb/Y3Qi6zQbOrC1CnOcGHvSUPPX3UCo+5Mee/NgmXWeu5rUezvLB5elgM453QNIBTNwuPYBUW4ZS
CYEzhsZDn08NiFsvwYLQRKIyOkZtov0il9Opc14jZqTo0DEqnXEY/4dXN/dRDiGS/YalqDrph2G1
Wsr6TF9zo7Wl5NaVNpwcS4zTUjqzl9hqCd5sP7BEe8+j1ANwRF+tZ2iun3EO+WeN1gmOflRfNI9k
3QC/1f/HEr5pFXJH8XdwMx5RsatBQPC0d0ggznlQTwDVRWlrfC93yU3zuv1ECKNEULNDlsvwtjbd
+WoLJWmgNghBh4PCP+zRbQJEM7zsWuf8suNrgfd12uXwKmEAAAPXQZruSahBaJlMCCf//rUqgABx
44jFfc9vrQ8CHU+cDzfDpwS4Av+8gXZ5uiW8Q+iQ+IjAN/bzlus5aO1GXOpZ+CkYTCNRyFeZXSku
LZwKfUTMyDmFUsH+EpzvDLq4PJX0+B0N3/oQoO3OVkQbG3QJpQMZ5AoKuooIzFKOhqB24j2KjBMl
n9f5CvUhWBTvj4ZFA1DDpRwQRY6pM6+an/8cUQX1h9g8a5WpoOwF1m6V5R9Pp9ogKYWuPP49ZO/i
Ek2m0kaHkyvo1dZAKhGZldc1DTjF3mo317xS82kuOqN8fi2pmut6xekedrgbLWPtAdhEZM04YduY
lx8sVfss9PpGbWR4HmjNziiBoDsgmziVDmSSxvbwcczIAn33aHowE5xoO39gUr7mQP7mmphmaPYn
wHMfVaA/SA4pwub7k2egB5ir2EmzdzA4wsnz4jq1MRfI3gkI3OIoCjAXutKRoHLNlIGd2R+kqICk
SNoi3324x9sCtyb7x6/kTsEuBs6EAcf4b8Ne1wpi9kzGlEMV44mqKS/5Uj4KBncgXOF7OcnrcDyN
Yni62hEg9q43QIt2gRD6x0ZC+MkBuo4R7M01t5u4qWSh6363rHWjrXqVh2lHujhwcd3XGxplyqVP
HxMGG5CVZJSWI37vrFNt2Z86FV16nfJaN6h0oMW8aA0RM20hX+vlkZlJd+7RsqGEq2s4cuKlNSSf
3nleZHWZZYReh2gAaxQUo356HqtuJ7qyIQbMosXeOo6u8ViFJHWSlffH7H6tX/ciJrerbMiV9VNT
tqnsx+NqrgFeaW9CKji60qRhtRZqD/B1V0tiA42SzNUoV1LWfb9pJe0MsnfniQVufr6pv+8vVFOQ
e9Hvkd41o5ki39ce9rZMvNkTDTpe45V79gRWGV7Rn0aohjBWMLYpb2IBfN3IDyZlwJlF1VHnsQed
lXqR99MRJgdmv9f/4W+A960dw7ITbfPqgwdCvd8G9faCLaA9EY3BzyNn0axezr3VA4rI2IQEQaSj
b7zbhSk2ULPYLFTHTSW+PY+hNZmV10a4AQEe9tPU1OxBx9ryugM8JXo8096RahIZCL6F8Il6WVzV
h8zg3CibuprXqkWlwu6qfJBkzcmxQ9gKQbrhc4vRhz3Z0PJRjP/Q80nevkGEKudUPwoOLFI4yDNV
t0zZOoYqukQzLqy4NEonYFAPVKw2ilVB07DMj4k7qS/vfe9EItCv+JIEWV5+/QpOlvAayTHvQvaO
GuqRP/3lmb5PFFYq0LtRu3Wbzc4hy5grFwBmn78uLCeAsW053AX8j6eZZMcXGVkq2/Rej4AAAAJq
QZ8MRREsEP8AAI6n6rfNnLLHL6CQ46IaZ5GVXO66d+L7q/Q5y8q9c0AyFGsC/kHb+tpl4ezjd4Y2
QaRgKW4dgqgO9iuG0HNvg/K46cFaApY9Wz39/pBgCWOp2uooeQ0nnkzSqcCPb1x9S1RW5tqqFFUd
6UR/qFGE43FYdWELPcXapCjKZ7NTlA/2BQG4Pe3a772zI1931Ab8FaumdMdV60MdZ8aBoiqJ+4kv
PqYyENE/4nlWpFeSE0VCdFR+4MVIx+b0+noakZMl0LFI4KmqdQKF9F4mRI1aLnK4mPonGFpuHCU+
W8pfe9tRKVx3+fimvvl9FGIWqrmu8q37ZFkfK+miVs1xsFDzYbRzdUqC19wQxr9HqG9MxhJOVjKp
APOlyO0+CvCubgL0WeL7nZwT2PBHa3q2Erm3GKZ9xIfvRA/h02juLkZZRgUD8jpKZ7Zh+H5MPuVE
9FhGvp7qx8DTXjTvdC5XIIbeIs/VWMlRowlhOnPpH/q6WQNNCZRfW9/bJwksrzrwP3AtIHv4K8ij
oQr39bbwBJwulFRcVbDqVtVvwP6Ac6we5O3n6eCEDcLF92kHSEL5DlONpVNvwQQkS+2bG18LQ/H7
o2egt649Lrzr9jt4K2fiB1GwM1dIVsg6ThqNvjRW+rnDdm8avKxWwxpUUn5jTx5iPLrFoSE2Zzlq
PCCt8srOhDD35EZo0t8QicWDE8rBqJV3xK4cUL8vzbljaIgmlK1Guvtj78W8f5/lKDYLBRcz4cVX
xxLCqDJ0GDiq7oyEjZA1Uq1sjHYET2Y34kBBfYFXAHRK67iyYFNmgoY7M4kZVUllAAABuQGfK3RD
/wABPl51vTNXIv/Fz5O2JGpFeG1s3rKB1Wy1jNsHPaLGly61DFRH30fGAisW5lL2APCk8f3u3PSC
m44AjbTlZ4NKpoIWvkicjYsDRilI9c3U3GpditHH/sWyZ3QVBLn8d913nk1fAAi11QSzX7HT/0+I
bXK0yalGamgxOppLer1fJvYJB8ItI7F0uo5YtsXqiYj6QPFURUgAnVA3g4qCYVC7d3NnFGvmzVBu
6z/LVaIO2zj/5wRBesSAcAGvB41Q2gKSCX7Gc5lcfYXnd6d/284zeGEQfAHmVKZ4He3uAZtpeOxJ
pr4z3XgB68hUwCRAuUGnLOrATccbY+yPMeIkypKXMuFvVkr+/DE6TP/ZP5KEmxU+WSdXt2ljxNhs
H/Kd60UhWSrklrHF1HyDR97PlF5yDVvjNr9SEbg+Hv35a4GdhmvM3ThnlPp6Q5v4fYg8TBVpRMne
4CWGkwRL3gbPCeAtg80IX3QyjZC3yWAwl+1jNlQQ4iMbItQLIj59TRWbiV8A71ZvkOs/8WJ+srcx
CBUWA1jCk56LVi2Jb57XOQZTbTnAri5K9LzhNOQjkfyvIgiHgQAAAdcBny1qQ/8AAT43tQYcQLh7
b8IRbf9TLgCMHZIyJFxNGI0ig5+0P7QLwHb9t8gAQ/VNCuWSzbo5qolkCEkCZ/0MBXwXcQGzD+sU
jI/pUIePDDb8uTuXliwCmRLzqHfHtborhcdOTRYCj/NvFxwbXr+aFUGGUs1y2ZEs/0OQj/gECcRo
2x/IsW6vavZgHpoU8SI8TSGfBH6/VBRxtCdHo8fdy8y0xXcteQacRrmsnMJO5u3NFNkIphWxic/S
WGh3mx3B7icFOdvznbLvgIYR6z9+eMxed8yj0WUVCKZMNpRIXE6LDAp3Q+NaBGG6HcBWNRfn5ie/
/wsLiXeX0iZr/7Sge6aok3/JTHqntJhmQ8TeAcD9kD5ul8ZhZVQockLcOQ3txcbp+QAYr+B6UvZk
2VX0nYcf3XQIeUNBgU+I6LZDYhs3/KX0m7Qa1kc1sOR5XF+WgkkFN+MsUkk276jFJAR75pt2d4bm
7MCgj0hvJbRKaAS3eiqzHl4k2pGUSEE4WAnFQFXQVZZUHO9LDKLemTbDv4666oBDV4do6HIgRsov
ol60dlFi51IqeSgxaNDZA1xVfZnSFW6Bcmo8dHaQUDYKRGvnEmCIbZ10PTZEjnl8NEf/1r6tA+cA
AAP0QZsxSahBbJlMCCf//rUqgAByflKO9dRAGUynSFTByfZz0C8a/kWAIlcDsMT4HzIdORqS8mg6
ej5XH498Dm/56owaRpl2zmS3O6E+4cmyPIMfOVWoEpVyriEDaIzjnmMABP+MhkUmQ6Gy0QtlFiil
7+yTVDzv0ALDqADb0t2ELFtBYxApaSx1TY2ixnwQmE9Z9Wkw69GGMjPl03Nt9LrGpkW+Qr79EL34
Q1qlrLUlwKjm0jYjJIXLo441oY0lYnA7rWFpXgwsw2hDH064kgSry1PONGJn8AErxpSO+yotKou4
KdZSeuSfJoeelz95MzRK1jA+iLF5k4+QQiAX1zC6Ppf53ow2D+3+PlscqMBAaMR3pxv6/f8ZAWJq
O88SgtPXi0rPeyhh72aJvKsy52Qx8d9JpLpv9A4yHuNp0DovX6cd7345B5PR3acCOCI62qIxaEeh
1NON4cpoGd+Ie5Uo71WvgVYquE1znPkZAHg2m+0eIrDwaGgTMuop3ypwGq1CFqzAf6xaszSIsbCT
TqaIevEAsOV14sl8Jaw0BlEOTd+nBn4DjkYrkvcSwapsHIwQPMObOLZkhDyB3TAQtpKZLg0wOPFs
gUZi3qL5Y7vlKbfX4upuJn68aqOo7/sB7y15KyzhuDUSJFiueXHbf4Qq42AhOneiStO46J04VxU6
FycYdgXbX7QSHTfui1+PJHNA4Ks9ZhZNo3H0sVu/X0hobWMpt6Fxvu+TZSd+KCFdVqBoiJI2xrVu
Ukuha0xLdTaJxyC9uqKhYAOUm9r7F1lX9IYU/G3rXGh/+SsM6wv24z4pbP4ER6NFybmOY8PIRyhi
XjRMoEuGdFfhfWg4G3kfUVc0cJHnVMx8tS7wXb22ZD81vriqWNQAQKKZfMcnbqZ/xU9aQWfLPPoE
eJMQdMA+MX+qjBdi9PnvRpsTTvozj9eRJNGGfIQMrHXhKdKR87tQEV/AmHupOJKFsk3FRfZbVrLl
bUMmBUCF6IA8a9MQeTvOAVYNDyrpaDYoH7DeAbsJjIdVqPlX4ZWaErOT77BMzSfMR+sXL/uOTYUV
iYOyVzz7qLtuWCnaKuiiOyrD95qIr+rak39pMxI32lO88yOfcAizLFyy0CUWmAfFDKuCLrk5va+L
u9j8x52l0xBHnpRVOZPSePt7PR6Y9ALhDPdAWbIngqShd800JoQXJcm2Nz63q/E4XzUu+xFUfaHq
5sPHx5ibz9dI9a3NvukjzAJ/A3TrKlvF5MGYnEOmvj2/5bnt91t17nj78rklN5z0PRmFhAs1TX7F
OMlibN8LRrTtjnrmTi3RaFFxHyP3j8IE5/8tPO5Bap8nCHfo7izhRXP+JEAGVQAAAjlBn09FFSwQ
/wAAjrbGiIr7ubwSfwhzNPTITL/NVd5FidbjRk6DUqZt/KYldXZcgCE1/qUBhlHwginfhdtk/T7H
0mRibK7P4zqd+UybLgGcurLuQctG0VtoARbkrIvYfLjHBXga7cFM7GY4605TirYDCuQOjj1X0tJ/
riKTLMPsf6951UjfSlIHEj1cWbOr9UN3TIOoBJ5N0E8uixSrLCB5veWyPQ97cBSr0zTPnQ3wjawv
Nfsp1C3FyUAf1h5q6gjYSVKecMa/to1tB5yikSQGyimUxizWuEVgxLdmL4igO3uF7wrwoK824jcg
GL5jw+rOcBABR4bYAfmabMwfbWOzX2VdF0Cxz+xdZN/6jrPd2j6PNAknmI/Z8BpaD1PyknlNZCC4
XkLbDDIudQw5MY3qsimhagIldCCEC3QsujNZTD1OmBJlerLR13hiGSXn8YMAk+sdrD/MtaGsyReN
Q/71LVk+LpF75lTD7rPIP9igOL8wRIHtDDOjwtltgZLo/DiqdT3uCPyk+Ah2ql87kinR0M5dWeHT
1ypueqVuiTZLahlIC/CIm+V+kiNFUnZYz8utfOR67mZTs3/uvDFDo6wFLqHxIlB8mCwbGVGvye5r
HiddqMiGfu2Z1SSalk+49e2qzvenELeWGC9spULf483CB7Bym7rgvOWC9/WO0qod6Oe2YoWTqTAt
GcojA93+tok9xkS1Y+hpyV1+LICe6DMQk9+CvdMWAKEF8Oc5hET9GD5dh/a0JHflHwAAAikBn3Bq
Q/8AAT4nPsQFyMJsTKHKYtrJZSiOWdGZoYHdSMK5spNDa4jaNd3EAENhwNh39KMDownzbU7mhCsz
l+2U5dR2FMB0OA0rQFBTFy+AflNTUzNCpPLjg2WC7fejBbhbK3Zjd/AfObXtNPjXZCz9hLp2rZYK
9gMJ/bEHgPY4oTolQW0BtvWy3PxRA60ifc+/ylZP0u9p7Lol7Rmgz2onRnu42ELYrrH+OEVQIul0
IW+KJLSP73F/I/cWxmI4Ref9MTMWJMKAqRGzkyhsvwCt9Cly6OhCtHE8ZW31fIVwFktVHRExb/s5
ZmgOkVeiEu/kK0OXQFzqNoMY/JkLhxj+FKSV2pLCdKN4Dj6JU34pQRWzP9USZ/MRUYSFYRAfDxDI
GNZa0ic52tMTOia1Lbzdn98TFI3PoJvRsIjpNwkqOD7q9ff4F7lrsp1O5Os6kEYXcffr6GkPmMME
MBCBtoN5Aedlee+wMDZ122hmdBnoUqP766ykwEABooo/pA3Zjh6JqXxrzOa+mB3XW3Y4Vi8FY3r4
DAVXVpLdakuzUkgEfLctiOjpagcd5oZAZ1rNe3OzCsGzNzqCP6tFY8+lpQCSIse2bsNpAl04UdtE
7lERLf6/L7kxhhs8nyc16lpqg/Cpo2+nYILO2Kuj+DzZcRQpI4yVi15GcHfL3p7X8SP/eqRHAlZQ
X76nV2B2UB/L7X8M0gDCGFjyzNE30cS8mfUvSfBw2qdoPHdMAAADm0GbdUmoQWyZTAgn//61KoAA
HJ+Uo711DCsT4XfuU0t9TWzDsLLfDabl6+O69gu7ej2NgCpM63gIAnEg+q2VBjPy8ms1EAk0wP/Q
eQdkaEc2/PlyOSA5U/8l71mFIUvrmj+KDcEv66KzpL20f8HcAcmxy+XeCBMLgPipe9Lt6UIt7BHN
MlvaAcQoZidR0X0AlGhRbmhWUqWvPXHq0MplFKOLC3SFVv01Xfqc0xeXSXFaAVf4AfzllB4bVhXP
8y4Yp0ujcB+wA5oGZqNCnI5/hhZkquIE9H/OJOZ07VjjiXTOSNt7DjUliZWL6PnpTDvClxZfd39R
RT4ycFUn52qluAmRoKd6INmb/MeIkL3R463FMUjzcUwhhDtvMBN/83oEvPnYXgm78JoVYH3f/Nnr
RUV0Vahs0ZfAIi5l6+5sWxy+6xexpHTo1RlSZgTFVFriG6CTHL5xWoRdVpl1h63aYx15EwvCC+GW
YnZ5EK+Q9D4DVvTq1cDN0AYR19U5x0laKtbg3qSsJuI9YRpO+dk29rkSgbBiKU4IKyV0rqWEL9uM
vHSIoChNwmVj66VyZCVXYS1nnSEm4IkHJXrXiMIgwaFSS1kohJ6N9QeQh2aOKoOBwqZngvuso6m+
dSszL5cemQ6HkodsY3kexMpBNxAmj/E52/NEkfj6J5S8yjP56p8Rs3CGyrUMeJECF68kOmRZSQ9v
3ir+n/+UP+Gz2h0xVXhoRgayAuoVa1DdsmHbiIWSHcxCTp+BWMGJVS2JAZ9C79iqDOaAuxnZfhaI
2o0ApXTy+pSjwAIPhku2sEh3m6/28clYtai3VrZ+wtbkmnmbxcGikHnAMJQs1uuhy7X4BIgX8VL+
96xFYp0eE4Y4ZBCMt72BrAD9deJfjn9ij4Kc2sTnjmQvPLiKfTjlxJU1hrdA8mBRykq5sKCNSdcf
hAJdCeriK059jekZriW0bjxHIfNLcwLK0+Urbi3i4zj0eOOnZre4DAWl+X8+7IAU0qtiYhtX3xD6
gSpdoVOrP+5J0fAtnJm9l/EMLdZHQMaab4ioO+x3TRaFC9kG56s4bjRzl2sSsuSbDiuyuhU3isfa
y/FS/v3XdB82ck1QG+DpvLMmHdoCB/5SqgXaFCCnFT6pz4//RneeKRVNUlrb3xkxvkD8k06VgbyK
uGRRA/bteAVOXz93Tz/te6u3DkjNp+1IOFVWURgP27BlXB5+0qB6RqZ1IzeO/TYgACDhAAAD8UGf
k0UVLBD/AACNVepX73y0fAAZxDOwp1GIh4G+5HJ1pD+6lUXFmRZA1/wwGNOKT2R7LJlFZ1XE//3N
JAYF1o8/44W9mlVmxD9bI5O8IcbEUhTVPeEX4cO1HVIZ1eWuSGFTTiy8HA4eN84td5d7iSBif5DI
dCr3JsDH2Lg84q8w3vKUNO8h3uqoi9gAsK8AWij3ixAfhj2f4SIcZxlYlWeoQ97ZX0pdWNzSMrbW
QScC5D22z6VTqsjlkwnp74iMLtsspw7XiGwyMMVruzM1bTmdDQciXxZv7vraZjQFndU01mv61ifk
VgQnwLSJy/SjH9knADQNbeRenCyQipA6IRn3C25PadpTJ9D0TWOABMWuVID2to8rcrERmuG43//x
9IgU9iRBqEAHf5F8f5HfApn9FfR3KqmxEwAVJpmoD2geGQhG8+l8OeWtNk9G5w0RRm3x/bjV92AZ
Rke+ZhkN3dXxoSM3yAfgqVDzqTBkAXRw6V8lM3zDnXMzFS9hv4qHWC3UvLlwiPB0hh+65pwtHuv7
7gvFV/KiK0CcWsIkqGXkdXJCnqz7Kqidvu/iQ4tqbAsK3IorHgapYae73ciYPoOUqWUmIY+MZGOS
LvQ7DSOmoGFtLhcX8x3c0ZBRuzzjEfZS1eQL3/BNNYDGNkKuOnmhzxg5D7Y+q5ntB5IiFFy95bWk
I8GHskzng1/Xud+PhD+MLEHGugG8KdGxZp4frXzABniSjM5tSRHa9ebzynzvbgaPMJyZ6oNqTrgg
G5kpVPQdIfxf7EWfUMGO9M/PlkeMb/QTk9Y33dqOoBdMf6V5+rU+LLArSnUFATMum2PNw6taiHqj
xvrbO2FcA6kCix40PATL6gRxREAT1K5T/fpUWJ5MOYzRQpfAw+kp6DiWCwkP60CL6gzzPw5k+MwI
3H6XncE7djZonMHHRd/fHwuBOPL3R/+3keVFj/DNzlbYK7DupGK/09mi/FCIQXkxVgIByiK0ZLj+
GddC/vzeQqn3qObQk0yVXWWOL+u6UBzgLxVkoGH8W1K49OUutxJzO72j8SJOplG5t21uoLWr7Tai
pdUI9KSIHWxsksibNh1C7kZcoeAZmAIwRr7Kx47M6klva3NXuHGAK3dW9MAARfHe56boNfIB9g1c
hKfsrnWzrr7UBm8vCQJy8+F0Plt9zqZQPkZR38EjuuLj9I/iM85/PNytWJNpqwR/xVN5jThdAaAf
iBxQT6FVHiZhozXFGOs9kK6yBjbAcO46HX+epiKPeHBaE451ow3OA8U7UAcMUfrDe8ckWUXM1Bku
ENubD4EH+g4nhRbMxmKJbA4nbYuwcF29QS1GEqDTEW3bYrH7f3AAAAF7AZ+ydEP/AATWM7CgVtnB
M2bAayFfOMOWYWgtKIIH7GLfIZNZiEtkSqOpXw+I/X4DU6eAD+gBz/3+COhkWEajzImbAqLywGRL
SlaJc+Pzm77LvfFvn9SheQUqx/USAhcWgxUWc8Xg4dTsBAJ+lhmrypbiyZo+0eTwkqmuKfrWLzcl
McZhsywcXjUrIp5/i0Psm7j5HLZpTQ0iVVFHgQYfO0tdKr18VA/HbFpPTQ8IBE/dAI4U6CyAJ4uW
WgR2FrMnj2zG4KovMhimDBG2XhjNaT9M1Xxj7oO4gpgBCwh2ULnX7olxEfg8Rl6Uca1n0ZOhQy2G
ZPeN0Aj7dcxbVkNkNXRq46j6Zc3oZbYL7Wt8ECE8MZCd2hRP2Zj9tXxMnJ8ZMw7omPTuaWLmvzUz
wC6thmJlyZTghsS9imPu4PsMSjfHNgV7zBBK7HvSsVsK+VzUuJC8tnNmp5JpBbd56zZW0vJobm38
pUxGCdLJPjez68suLUqfT9eYFP6ygAAAAYcBn7RqQ/8ABNWkumn1fz82bDg79zp929KcJ2WQp0XN
bVgo/AMNpg0BrFhYol++3ABqj4exYoH1JMVBQMdvWtgkABnx7aveO8OMz67ZWQsd2a4Nd4M06n6z
HB7VhsEdSaE63qJB4nLjHumPv+NoAhujl3MLg3sD3b2Qit3cRFuFXX0ma2sMpyrAdByDckKhWOF2
wZceQ+CzV9tTXUC12p0am06b+xQ4oLjm7+pE0HCZFglJrlnUbqc++QtLrXxhk4v4xLIUhxDKSsbl
CPXb6X88u4GUueMD2aLduuOtqvKtDqxgSEi7i2Z9sP9xVuVCm98ipTWM8N5RrUVGJLXTKo7b0x6N
nKZDztoH+vuaZtb7d1zBKmPsdXNUxEuujvkI99K6ureqbc94K2kOhh7URZKZlh7bDClAHCZ9FG6j
KrIpne9/GfMhzoP+EBk4uzV/GRjRTOih1NhyXeGovH8c9Zpn7yGxZNUWrP5x+/oS3gSFMJcoDRgn
mprcVryehzMopKx8/MG23QGJAAADfkGbuUmoQWyZTAgl//61KoAABx0s5lktMo1bp06XbfpOoKec
G07B5vpy7s1GwqDWtWltaTKG+4QQppQDeB8mLRAFHpWPFDpp9JYWA2z8M8ZJodMcCMiP7PM4rNle
9DPcy1PdQW0Om86tcmskDkrmdLZiTApXt5bkZPi2yy1mZ/cPK9sP1xHr7bVsKk3QiFRkrNRDt7gH
DLMMrJ7vSu9Ac6Wmy7H4T/UxZhFIX7m8Zgr2MqOBqxF8kNy5ajt9n5ErMDpyTs2xeFqc0Q+jl7/J
ahiz55KpQizK/ffaBD8cgp1kb01qjUQKMDOMwBkC3U7KYVDvev3/aemoesP92S69Pkn2xI0n2cxB
pSkPmOdUU893bbOX3X1SS0qX432x7k8XkzHJvyBZpMXqCBhZJ9Hv6ep1G8DwPanJpMJKv0HE5RX/
r74ucShFjOBeqvtEGnOo4jVqRFobEdvufQ3Cxo6Mb5g/55z+cUkCXIx/qfbTZVfgKatTFURJxFBj
wwWzLo89Qgvin5YaWyDhFEzZB2hEV84Un96J3fOl1GYbwy3fazEO0eF4F8AnzB0DXS+FNCdFRua/
M3ji1vSNY+q1M0wk9RH5lObBH6IBEEz6PWWsYq+hVsf1jiEBZDv/4RcLL7USkf8IsagGIaUedXuE
8LT4vxNZmsKP1JwSMPFeYmHb8nStiRAo6ljKTcgQqOj9VBcVeqDY2HVZ1cZYfD0Ud3jwfIHHnvPQ
KJPr2Z8kRamfO5zbZ1avD1xPKD73r9HobRgKG0SIJtlRj/OX4OPSzDUmdqD46gs0DzLOTQTlYJEs
Afvs/cciBJqf0B1XaGAJP9VtIja8aGlnS1cLoJOV9pX2BbahP9e7/a7BIBa649IL+mOhqaQXE1Zu
gtRvhBra3kaJb7nljS/0XF3pqAuC1C4iCahmF0UgsCkYckZO5UJrvjKh3O/koasgQwWJ728hAGFK
T7dFmIsBeVtWUE6u+OgYJ2eImvPM6bVKrwKLgYmLiqY2Zw9YaXEV6WzjuBjwbOj3RN2Ik2zdK6xw
aAG4mIpq58IeR/xYsgRJuVTMqVdQBoRXUlVrEHfnPLvs9WE7nNm7yeG4wWtogSaGBioxhZfRrhKl
axUDAKd+OYKLgvdifchQXSzFpIMaDW8V3wTaThjCTnKKlzkuZbFEJMSqSR5Bld7T6u8VVSiDR6/f
IWCCmgAAAklBn9dFFSwQ/wAAEdbZIxBZ99Wv0D79OWSC2inVXWL5J4Vj7RrldrME29fPKkfAc/LJ
Up46k6W47HXh1R1T/8V3gA73BdxdLEbRVVq9hIjuuzOXG7AesPrLseWSs1gTQMeZkq1VjOvVUOYI
4rkzR+HDLUDKzixMs5POqVQ00BWYQLyGMNKAa8IYx/NhtBdEh3jV6k9LFqbMTW85SQl4bXhKWTeg
aRWGgZ7xwwGCQ8Q0B2apreLfGTIXXN3kbTXTjMIPet2iH5Z5O2es+WxbKmIpfXO5nhX7ikLln/dA
hBOR4Bq6M6CkyCwoiwoGtFdVqNrlMs+s4VGDmp34XsQMVfdS2Cic1pQgBqo3lZpNO4Pmkp6LwW/5
308eVvaiZWFQ1s3j1OC8Ivip1cUIVYGlqdqUqGj6AvTzS03a2ZqH+SIwqUYlAaHBfzUGkMCI3QU0
OzEY3cdG3rV6EEG+v1wjVDCwQwQokvNzGJG1vJgC1Xl1IPdBie3EYVpYhGRF+YvIYTcAsBu1LKJy
OR+a1uyeNyCnMLsSU7eqnrheJHMcvTWjkg2uBR/9iosW2RL8rl6yCEC4bH5zF9bKu1RpuqNbb0Ct
LjKQ6GJK2INAbxG9QMIZkXuccvdNRxiwY2s9oayrkE4+6xjUvRt0wIILh7W2bLu/FgQwzpm5u36c
7xfbq4Leik63zdaymUPvVXZuljT2eNd4MsQucafEqR6ViRnla/ribrC6ulk1RPAGaxeJNYCQc21G
GheEtLII3RfizJWKs3KmPcay/kVkwL0AAAFpAZ/2dEP/AAAnwuB+lsVW7KwtkJKs9HmnHM2VVZoq
bktEIU/6kseqzo3kPZHWfNIt008/zewocWvdDQLEJVpSJ5p06nlhH5tOuWUloTUQkqChqFx5Ilyw
ia0l4q//AQP+7YAEJbxeSXsGCwYwHD4tbxZXtG1DkICWmbDYYYUUx2lVXFBR6Wj9NsS8ZEKTudZK
pATQYjwJh7LmnwlUPJeHDMz5TFv4c7xUN1x1zrriV5gwdvUCYOWqN3spcxmHONI6vgZIOmiSycUk
Oy5E0mYjLKpNBS5c2va1YrYoZUWwxW3qFGXvVYrhfylZS1Av4qP3xUAEXTNYo3ta1fQZDUyPWn+l
0b9TQL8R0xxdyzHapwnQWj/ReaI7T5oFDbBI33XOdXi0uktskUVrkPXSneS0QBWM5DyVB9kLaDOE
kzZnq/VzmJXH0WUBDebJcs132p4PpDLpH/eybA3mBmpUJgFOAbWUGZz/t57APQAAAacBn/hqQ/8A
ACfI/FVkS9VeWuN7KNmuIJZXjzWgO91gbEaFXyRKjxYvCcjNyEjuFkD+jI7C1cjvZwSY1GTIoXf4
VXyFISFYm62UzgAhLqkoA5X+ynSudCmVJv9WuevRA+GGUL5ucXG0qMGbvgqIa5KmQ0H3FZFOCh4J
IobNRw+3PXKHs2hdPCim7orrcLJy9Y0bDtUPI9RulkDNq0ULqp7PHBn80OUkWrSglwSKAqbzXG8v
ev5aGjwfXE7WsuwYGwwjSTo+T0iKwlErLodZDd2aSVMEt69DBoPb9BxCPyP3+RUH7frRndtyvO1L
ZIIhCBn+P00SCesyzlNYP26nu92BZip3u2jmR6PHLcdkPtgv1hTkPLfjxqFmFG4Er7wMhSnRkswn
9y6/qQsJkjyNdvHC+8Ef3OESTMIoxPPgQp5TwG8mnpMZFOA38Vy3S5b519R3ftAmOa1M92U29ka3
acm6ygG0po27MWqvlN5AehJWw8wKUwFsq0Ww+6HjZO6K2AMP04eJ7X/LQ/JS4NMS/Jevfa192W7x
3D/al64HBMzQUB7BFcj062AAAAImQZv9SahBbJlMCH///qmWAADeVLZ4Nyb2qwlX9Qbo7/P76U1x
OzhjElFiXte0hNlOAj/rUOuJ96CCsF2feRFgWFj0sTrmuznjzF/jvahSQEOM0ILwUyNaiEB+0LdK
pfo8X8UGzWYKciAhoYDqkqRejfTm9MNr9CwRee4oADrvnaRAt41dt5ZqFGNfuAF2tjL3d5AfqQ66
miPLiSR0o66RTBPF0KakEaHWK0a3yXZZehBflExKslGF9O13EwY3OzV1iocdAcfo/VGPJPWiGSgL
b613wIWIEV93TFWBEj24HMtf+gAUGqBtx03zU2jv9VncsuKAtjIMJXtBjHcG7IPGvffI981CRjaQ
zkagw2ZSh93NmwSZLRBTacGeps1EywS9knLs3NRbuCTc5NukJ2lqSWB0uRGH+qa/e14VFN75ToUd
22P+seghdR77P58kYL64AmFYczfbrN85hGhzl6aYegdd+1OcY383/LQChC+h5auhK3Sa22jppjOv
ODwbec9pMI2x9aETP9V8MT7lLYwf4gAHT7nxIa/FnjjM5E4JopvXT1B/ElsM1XOEU7LOJ0joNzL+
HoKY3Yd8Ia3atcPxCkScBnfa+4YpKNypY7dy6jFCnxN9LsI9qSBwKcHt8IbpBgwssi6bYaCPf//m
Pxch+hLiCfn612lSwoihG78tqN8YDzNE8QTlQ1n6v74nVp6LLUoCCQqFTuAsAs+93ubAGdoSe+AC
TwAAAa1BnhtFFSwQ/wAAI6trqu11p+1dpEKsksEeumkgC4LoYsH3y1DI8+e3GN9eAEieF0QuheTF
ao/QREgoFbQQkxlphAfO4j9Ih9ctFFiNtj2eH3AvqMyR/gfBZCUkXBKosoqeH+r1PCw6GvyOwx19
dgCGtRrIfFInuW8HDl8m4QHcRSwS4VJuEa3IXkfQPMFbq8HMPL6ejlJWke8wHOMLEbefLn/XJsOY
2hVRFjgtTpzTKAEw2a5vg0yIAcOrC/09cs4mo2zEhEmiIhGx1Y1QGZhx6yKf+g9YzUkdk4iLEr+z
BttEL06nH/ohvMe7DJBrGKI4pYre0eyqnI/U2czTCWvnxBRA5VkFdXX+gu2T/tSomkMeDMvFNSY1
FX4Nu18hAqb5Q/4FHxGL75LZ244/kXc7OS9aC8/6ab7GDVOVtPMa++WF9EZfcteOJKV1AGCHgQlc
Oh/61RIpCFA4UAu6+Rn/4tls8WtNkndWNv0Nkkwt/Y3qX4ij64JrJRAIx8BX5Zf8b67d0NW5e4pl
n5s1yMtFeJb2PQUXrG3RIPiG29cRbwzuqVQuSUADIDzZ53QAAAFwAZ46dEP/AAAnwuB+ls6u8TAx
GkOGkeXCCUEytPh0M27ESR6ixAF314kwUa4q1EbubzDskZIjzeEPkzemav62FmykY0m1loAHBDgf
Nh07A1w7QNzmr3aoqCZIzJTOCfu4V9BaO6blLewUbuA7TRdEFvkoPMuk1LilUiIE5xx1/idOzWs6
eJDDt3GFhAm5VmotWv8/pIJL1LuYQosnqPi3waTq8YPW1YtH5pmFKxLfDs1fOzRIpIHGpU7xOkBh
q97Qn0rgudpKROPnE/XOs6dwch052K/6SeO3XVBpYp2yEXsB87jfQb7emQ2KufbqlqpU4o6lytLx
xgfL2NJKF1QeunvQ/p9p9gptIS473SGZb8WWUyQAqdqdFBesW7vY5nppgLGEc/i2RWwouX/x3ZUH
mqiT0nbmVgLu8V/DqeshrhxJr2fQAgDkVx69kblehVVvi11Y8Gw1g7OSOrnistP7obi4q1KAKW81
ZISE8PlwNO0AAAGmAZ48akP/AABPjjHubmEdRaVxdlGYcJQSEz9fgPevogOVCsiKxq2peoANjQAP
jE7hfzDi0bkLEqwWECkjvaiUMjx0sVwBijk0Q2DwNmv6F8/peKqmKKyXb3YRMzDB+PWzfN8PJhz7
O1E7OmxUkgRPe5Qq+q7Yx2I9XyZ2VAfBvIiKCmFHEi7+mT4JOlicBDWn+4yhK3MjGqhoH+vesBNJ
7bbg/fIPFlEDdi3gf+KxlegyrS+q8JVEJrD4SQtwyeAs8C3RD0iDwZTgjiXC2r1rs4QRRIJzk7Xt
BLm/8AkQRR9g4VQuBM+DKsPDmlxlXz5wl3gLLP6wUclGvhoi8uu0Wl8Fey5ZEohYBCfvQhhizb2t
RpJeskofo04dLzO1kV86ypElqfx7TTMY3M9mUb8epXPn9w/DrHgQfuzUiPV2gdhuBjyTs77LwbzT
ROUbMQxjmikDBskl9YD6r8QyaJITx+9HMvEsE/w7R0Rnf4nKj4LdhtPLPAIIHcqpDxhufVUXaW0r
r6amcLVS5K1wb0ZgOy8PjZDVHkFvrJVK5A4SRr/+1yRi/8EAAASDbW9vdgAAAGxtdmhkAAAAAAAA
AAAAAAAAAAAD6AAAF3AAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAA
AAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAA610cmFrAAAAXHRraGQAAAAD
AAAAAAAAAAAAAAABAAAAAAAAF3AAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAA
AAAAAAAAAAAAAABAAAAAA2AAAAJAAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEAABdwAAAQAAAB
AAAAAAMlbWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAAAoAAAA8ABVxAAAAAAALWhkbHIAAAAAAAAA
AHZpZGUAAAAAAAAAAAAAAABWaWRlb0hhbmRsZXIAAAAC0G1pbmYAAAAUdm1oZAAAAAEAAAAAAAAA
AAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAApBzdGJsAAAAuHN0c2QAAAAA
AAAAAQAAAKhhdmMxAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAA2ACQABIAAAASAAAAAAAAAABAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGP//AAAANmF2Y0MBZAAf/+EAGWdkAB+s2UDY
EmhAAAADAEAAAAMCg8YMZYABAAZo6+PLIsD9+PgAAAAAHHV1aWRraEDyXyRPxbo5pRvPAyPzAAAA
AAAAABhzdHRzAAAAAAAAAAEAAAAeAAAIAAAAABRzdHNzAAAAAAAAAAEAAAABAAAA6GN0dHMAAAAA
AAAAGwAAAAEAABAAAAAAAQAAGAAAAAABAAAIAAAAAAEAACAAAAAAAgAACAAAAAABAAAYAAAAAAEA
AAgAAAAAAQAAIAAAAAACAAAIAAAAAAEAACgAAAAAAQAAEAAAAAABAAAAAAAAAAEAAAgAAAAAAQAA
IAAAAAACAAAIAAAAAAEAACgAAAAAAQAAEAAAAAABAAAAAAAAAAEAAAgAAAAAAQAAKAAAAAABAAAQ
AAAAAAEAAAAAAAAAAQAACAAAAAABAAAoAAAAAAEAABAAAAAAAQAAAAAAAAABAAAIAAAAABxzdHNj
AAAAAAAAAAEAAAABAAAAHgAAAAEAAACMc3RzegAAAAAAAAAAAAAAHgAAL4gAABdyAAAGCAAAD+cA
AARdAAADJwAABmYAAAIBAAAOSwAAA+UAAAITAAAD2wAAAm4AAAG9AAAB2wAAA/gAAAI9AAACLQAA
A58AAAP1AAABfwAAAYsAAAOCAAACTQAAAW0AAAGrAAACKgAAAbEAAAF0AAABqgAAABRzdGNvAAAA
AAAAAAEAAAAwAAAAYnVkdGEAAABabWV0YQAAAAAAAAAhaGRscgAAAAAAAAAAbWRpcmFwcGwAAAAA
AAAAAAAAAAAtaWxzdAAAACWpdG9vAAAAHWRhdGEAAAABAAAAAExhdmY1OC40NS4xMDA=
">
  Your browser does not support the video tag.
</video>




    
![png](11-hipoteses_files/11-hipoteses_53_1.png)
    



```python
#In: 

```
