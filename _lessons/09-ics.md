---
layout: page
title: Intervalos de Confiança
nav_order: 9
---

[<img src="./colab_favicon_small.png" style="float: right;">](https://colab.research.google.com/github/icd-ufmg/icd-ufmg.github.io/blob/master/_lessons/09-ics.ipynb)

# Intervalos de Confiança
{: .no_toc .mb-2 }

Conceito base para pesquisas estatísticas
{: .fs-6 .fw-300 }

{: .no_toc .text-delta }
Resultados Esperados

1. Entender como a distribuição amostral faz inferência
1. Uso e entendimento de ICs através do teorema central do limite
1. Uso e entendimento de ICs através do bootstrap
1. Como os dois se ligam

---
**Sumário**
1. TOC
{:toc}
---


```python
#In: 
# -*- coding: utf8

from IPython.display import HTML
from matplotlib import animation
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

    /tmp/ipykernel_15850/1539025901.py:1: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      plt.style.use('seaborn-colorblind')



```python
#In: 
plt.ion()
```




    <contextlib.ExitStack at 0x7f48e83c0050>




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

Vamos explorar a ideia de intervalos de confiança. Inicialmente, lembre-se do teorema central do limite que diz: se $X_1, ..., X_n$ são variáveis aleatórias. Em particular, todas as VAs foram amostradas de uma mesma população com média $\mu$ (finita), desvio padrão $\sigma$ (finito). Além do mais, a geração de cada VA é independente da outra, sendo toas identicamente distribuídas. Quando $n$ é grande, então

$$\frac{1}{n}(X_1 + \cdots + X_n)$$

é aproximadamente distribuído por uma Normal com média $\mu$ e desvio padrão $\sigma/\sqrt{n}$:

$$\frac{1}{n}(X_1 + \cdots + X_n) \sim Normal(\mu, \sigma/\sqrt{n})$$.

## Distribuição amostral e Intervalos de Confiança

A distribuição dos valores de uma estatística a partir de amostras é chamada de *distribuição amostral* daquela estatística. Ela tem um papel importante, porque é a partir do entendimento dela que estimaremos quanta confiança temos em uma estatística que estamos calculando a partir de uma amostra. No exemplo acima, cada $X_i$ é uma amostra e $X_i/n$ é a média desta amostra. Então, $\frac{1}{n}(X_1 + \cdots + X_n)$ é a distribuição amostral das médias!

O principal a entender aqui é que se conhecermos a distribuição amostral, saberemos quão longe normalmente a estatística calculada para uma amostra está daquela calculada para a população. Sabendo isso, podemos calcular uma margem de erro para a estimativa feita a partir da amostra, tal estimativa será o nosso intervalo de confiança.

Vamos iniciar com um caso que conheçemos a distribuição da população.

## Exemplo Moedas (Caso onde Sabemos da População!)

**É importante falar que por um bom tempo este notebook não vai computar ICs, preste atenção no fluxo de ideias.**

Por simplicidade, vamos fazer uso um exemplo de lançamento de moedas. Isto é, vamos explorar a probabilidade de uma moeda ser justa usando estatística e amostragem (conceitos não exclusivos).

Lembrando, temos um espaço amostral:

\begin{align}
\mathcal{S} &= \{h, t\} \\
P(h) &= 0.5 \\
P(t) &= 0.5
\end{align}

No caso das moedas é simples saber a **distribuição da população**. O número de sucessos de lançamentos de uma moeda segue uma distribuição Binomial. A mesma se parece bastante com a Normal. A PMF de uma Binomial é:

$$P(k; p, n) = \binom{n}{k} p^k (1-p)^{n-k}$$

onde $n$ captura o número de caras e $k$ o número de lançamentos.


```python
#In: 
p = 0.5 # probabilidade de heads/tails
k = 30  # temos 30 jogadas
x = np.arange(0, 31) # Valores no eixo x
prob_binom = ss.distributions.binom.pmf(x, k, p)
plt.stem(x, prob_binom)
plt.xlabel('Num Caras - x')
plt.ylabel('P(sair x caras)')
despine()
```


    
![png](09-ics_files/09-ics_8_0.png)
    


Usando a função `ppf` podemos ver onde ficam $95\%$ dos lançamentos de moedas. Para isto, temos que considerar $2.5\%$ para a esquerda e $2.5\%$ para a direita.

A `ppf` pode é inverso da CDF. Pegamos valor no percentil, não o percentil dado um valor.


```python
#In: 
p = 0.5 # probabilidade de heads/tails
k = 30  # temos 30 jogadas
x = np.arange(0, 31) # Valores no eixo x
prob_binom = ss.distributions.binom.cdf(x, k, p)
plt.step(x, prob_binom)
plt.xlabel('Num Caras - x')
plt.ylabel('P(X <= x)')
plt.title('CDF da Binomial')
despine()
```


    
![png](09-ics_files/09-ics_10_0.png)
    



```python
#In: 
# 2.5% dos dados P[X <= 10] = 0.025
ss.distributions.binom.ppf(0.025, k, p)
```




    10.0




```python
#In: 
print(1-0.025)
# 2.5% dos dados para cima P[X > 20] = 0.025
ss.distributions.binom.ppf(1-0.025, k, p)
```

    0.975





    20.0



**Caso 1: Quando sabemos a população é fácil responder a pergunta**

$95\%$ dos lançamentos de 30 moedas justas deve cair entre 10 e 20. Acamos de computar lá em cima usando o inverso da CDF `a PPF`.


```python
#In: 
p = 0.5 # probabilidade de heads/tails
k = 30  # temos 30 jogadas
x = np.arange(0, 31) # Valores no eixo x
prob_binom = ss.distributions.binom.pmf(x, k, p)
plt.stem(x, prob_binom)
plt.xlabel('Num Caras - x')
plt.ylabel('P(sair x caras)')
despine()

x2 = np.arange(10, 21) # Valores no eixo x
prob_binom = ss.distributions.binom.pmf(x2, k, p)
plt.fill_between(x2, prob_binom, color='r', alpha=0.5)
```




    <matplotlib.collections.PolyCollection at 0x7f49380544d0>




    
![png](09-ics_files/09-ics_14_1.png)
    


## Simulando

Agora, vamos assumir que não sei disto. Isto é, não sei nada de ppf, pdf, pmf, cdf etc. Mas eu sei jogar moedas para cima. Será que consigo estimar o mesmo efeito?!


```python
#In: 
# Jogando uma única moeda
np.random.randint(0, 2)
```




    1




```python
#In: 
# Jogando 30 moedas
np.random.randint(0, 2, size=30)
```




    array([0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0,
           0, 1, 1, 0, 1, 1, 1, 0])




```python
#In: 
NUM_SIMULACOES = 100000
resultados = []
for i in range(NUM_SIMULACOES):
    jogadas = np.random.randint(0, 2, size=30) # joga 30 moedas para cima
    n_caras = (jogadas == 1).sum()             # conta quantas foram == 1, caras
    resultados.append(n_caras)
bins = np.arange(0, 31) + 0.5
plt.hist(resultados, bins=bins, edgecolor='k');
despine()
plt.xlabel('Numero de Caras')
plt.ylabel('Fração de Casos')
```




    Text(0, 0.5, 'Fração de Casos')




    
![png](09-ics_files/09-ics_18_1.png)
    


**Caso 2: Quando sabemos gerar dados que seguem a população é fácil responder a pergunta.**

Podemos verificar o resultado empiricamente na CDF. Estou usando `side='left'` pois por motivos que não entendo o statsmodels faz `P[X < x]` e não `P[X <= x]` por default. Com side `left` corrigimos isto.


```python
#In: 
from statsmodels.distributions.empirical_distribution import ECDF
ecdf = ECDF(resultados, side='left')

plt.plot(ecdf.x, ecdf.y)
plt.xlabel('Num caras')
plt.ylabel('P[X <= x]')
despine()
```


    
![png](09-ics_files/09-ics_20_0.png)
    



```python
#In: 
np.percentile(resultados, 2.5)
```




    10.0




```python
#In: 
np.percentile(resultados, 97.5)
```




    20.0




```python
#In: 
ecdf(10)
```




    0.02159




```python
#In: 
ecdf(21)
```




    0.97828



Até agora eu estou assumindo muito.

1. Sei da população
1. Sei amostrar de forma uniforme da população.

E quando eu estiver apenas com 1 amostra?!

1. amostras = []
1. para cada amostra de tamanho 100:
    1. amostra[i] = np.mean(amostra)
1. plt.hist(amostras) --> normal
1. estou trabalhando com uma delas: amostra[10]


## Quando não sabemos de muita coisa

**Preste atenção a partir daqui**

Não sei nem jogar uma moeda para cima. Desempilhe o TCL.

Lembre-se que distribuição Binomial captura a **média** de caras esperadas em _n_  lançamentos. Note que, ao somar cada um dos meus experimentos estou justamente voltando para o **TCL**. A distribuição amostral aqui é a média de caras a cada 30 jogadas. Assim, podemos ver a aproximação abaixo.


```python
#In: 
bins = np.arange(0, 31) + 0.5
plt.hist(resultados, bins=bins, edgecolor='k', density=True);
plt.xlabel('Numero de Caras')
plt.ylabel('Fração de Casos')

x = np.linspace(0, 31, 1000)
y = ss.distributions.norm.pdf(loc=np.mean(resultados),
                              scale=np.std(resultados, ddof=1), ## ddof=1 faz dividir por n-1
                              x=x)
plt.plot(x, y, label='Aproximação Normal')
plt.legend()
despine()
```


    
![png](09-ics_files/09-ics_28_0.png)
    


**Qual o siginificado do plot acima??**

1. Cada experimento foi n-lançamentos. Tiramos a média dos n.
1. Tenho a variância das médias, ou seja, a variância do estimaodor (lembre-se das aulas passadas)
1. Resultado final --> Normal!

Observe como com uma única jogada de 30 moedas eu chego em uma normal bem próxima da anterior!

Cada jogo é um vetor de booleans.


```python
#In: 
np.random.randint(0, 2, size=30)
```




    array([1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0,
           1, 1, 0, 0, 1, 0, 1, 1])



A média é a fração de caras


```python
#In: 
np.random.randint(0, 2, size=30).mean()
```




    0.6666666666666666



E ao multiplicar por 30, tenho o número de caras, afinal foram 30 lançamentos.


```python
#In: 
np.random.randint(0, 2, size=30).mean() * 30
```




    13.0



Colando tudo junto, note que estou computando o desvio do estimador! Para isto, use a variância do estimador.

$Var(\hat{\mu}) = s^2 / n$

$Std(\hat{\mu}) = s / \sqrt{n}$


```python
#In: 
uma_vez = np.random.randint(0, 2, size=30)
mean_uma = np.mean(uma_vez)  * 30
std_uma = np.std(uma_vez, ddof=1) * 30 # o desvio padrão é na mesma unidade da média
std_est = std_uma / np.sqrt(30)
```

Observe uma normal muito próxima da anterior com uma jogada!


```python
#In: 
x = np.linspace(0, 31, 1000)
y = ss.distributions.norm.pdf(loc=mean_uma,
                              scale=std_est,
                              x=x)
plt.plot(x, y, label='Aproximação Normal com Uma Amostra')
plt.legend()
despine()
```


    
![png](09-ics_files/09-ics_38_0.png)
    


Observe que ao fazer várias amostras existe uma variabilidade na normal estimada. Vamos entender teóricamente.


```python
#In: 
for _ in range(30):
    uma_vez = np.random.randint(0, 2, size=30)
    mean_uma = np.mean(uma_vez) * 30
    std_uma = np.std(uma_vez, ddof=1) * 30
    std_est = std_uma / np.sqrt(30)
    x = np.linspace(0, 31, 1000)
    y = ss.distributions.norm.pdf(loc=mean_uma,
                                  scale=std_est,
                                  x=x)
    plt.plot(x, y)
despine()
```


    
![png](09-ics_files/09-ics_40_0.png)
    


## ICs com Normal

Considere o caso que tenho **UMA** amostra apenas. Aqui eu não tenho mais distribuição amostral, pois só fiz uma coleta de dados. Felizmente, eu tenho algo para me ajudar, o TCL.

Quando o TCL funciona, podemos computar o intervalo de confiança usando uma Normal. Essa é a base comum que motivamos algumas aulas atrás. Vamos brincar um pouco de shiftar/escalar nossa Normal. Sabendo que:

$$\frac{1}{n}(X_1 + \cdots + X_n) \sim Normal(\mu, \sigma/\sqrt{n}).$$

Vamos fazer:

$$Z_i = X_i - \mu$$

Note que estamos apenas jogando todo mundo para a esquerda $-\mu$. O valor esperado (média) de uma VA X, $E[x]$, menos uma constante $c$ nada mais é do que $E[x]-c$. Além do mais, a variânçia não muda. Veja as propriedades no Wikipedia. 

Assim:

$$\frac{1}{n}\sum_i Z_i = \frac{1}{n}\sum_i X_i - \mu \sim Normal(0, \sigma/\sqrt{n})$$

Agora, vamos dividir $Z_i$ por. Neste caso, o desvio padrão e a média vão ser divididos pelo mesmo valor:

$$\frac{1}{n}\sum_i Z_i = \frac{1}{n}\sum_i \frac{X_i - \mu}{\sigma/\sqrt{n}} \sim Normal(0, 1)$$

Isto quer dizer que **a média** (note a soma e divisão por n) das **distribuições amostrais** $Z_i$ seguem uma $Normal(0, 1)$. Note que estamos assumindo que o TCL está em voga. Às vezes (quando quebramos IID ou Variância finita), o mesmo não vale, mas vamos ignorar tais casos. Bacana, e daí? **Não importa a população inicial, essa é a beleza do TCL!**. 

Então, mesmo sem saber a média real da população $\mu$, eu posso brincar com a equação acima. Primeiramente vamos focar na média $\frac{1}{n}\sum_i Z_i$, vamos chamar esta distribuição de $Z$ apenas. Sabendo que uma população segue uma Normal, eu consigo facilmente saber onde caem 95\% dos casos. Isto é similar ao exemplo das moedas e da Binomial acima. Porém, note que eu não assumo nada da população dos dados. 

Uma forma comum de computar tais intervalos é usando tabelas ou uma figura como a apresentada abaixo. Hoje em dia, podemos usar a função `ppf`. A mesma indica que 95% dos casos estão ente $-1.96$ e $1.96$. 

![](normal.gif)


```python
#In: 
ss.norm.ppf(0.975)
```




    1.959963984540054




```python
#In: 
ss.norm.ppf(1-0.975)
```




    -1.959963984540054



Agora eu preciso apenas voltar para $X$. Para tal, vamos fazer uso de estimador não viésado de $\sigma$, o desvio padrão da amostra.

$$s = \sqrt{\frac{\sum_i ({x_i - \bar{x}})^2}{n-1}}$$

Fazendo $z=1.96$ e $P(-z \le Z \le z) = 0.95$

\begin{align}
0.95 & = P(-z \le Z \le z)=P \left(-1.96 \le \frac {\bar X-\mu}{\sigma/\sqrt{n}} \le 1.96 \right) \\
& = P \left( \bar X - 1.96 \frac \sigma {\sqrt{n}} \le \mu \le \bar X + 1.96 \frac \sigma {\sqrt{n}}\right).
\end{align}

Substituindo $\sigma$ por $s$: a probabilidade da média da população está entre $\bar{X} +- 1.96 \frac \sigma {\sqrt{n}}$ é de 95%. 

1. https://en.wikipedia.org/wiki/Variance#Properties
1. https://en.wikipedia.org/wiki/Expected_value#Basic_properties

## Computando um IC dos dados


```python
#In: 
# brinque com este valor um pouco, observe a mudança nas células abaixo.

TAMANHO_AMOSTRA = 100
resultados = []
for i in range(TAMANHO_AMOSTRA):
    jogadas = np.random.randint(0, 2, size=30) # joga 30 moedas para cima
    n_caras = (jogadas == 1).sum()             # conta quantas foram == 1, caras
    resultados.append(n_caras)
```


```python
#In: 
s = np.std(resultados, ddof=1)
s
```




    2.8306404315146163




```python
#In: 
s_over_n = s / np.sqrt(len(resultados))
s_over_n
```




    0.2830640431514616




```python
#In: 
mean = np.mean(resultados)
mean
```




    15.26




```python
#In: 
mean - 1.96 * s_over_n
```




    14.705194475423134




```python
#In: 
mean + 1.96 * s_over_n
```




    15.814805524576865




```python
#In: 
# até aqui.
```

## Entendendo um IC

Diferente de quando temos uma distribuição populacional, temos que interpretar o IC diferente. Note que:

1. **Não estamos computando onde caem 95% dos casos da população**. Basta comparar os valores acima.
1. **Não estamos computando onde caem 95% das médias**. Bast comparar com os valores acima.

Estamos resolvendo:

$$P(-z \le Z \le z)=P \left(-1.96 \le \frac {\bar X-\mu}{\sigma/\sqrt{n}} \le 1.96 \right)$$

E chegando em:

$$P \left( \bar X - 1.96 \frac \sigma {\sqrt{n}} \le \mu \le \bar X + 1.96 \frac \sigma {\sqrt{n}}\right)$$

**EU TENHO 95% DE CONFIANÇA DE QUE A MÉDIA ESTÁ ENTRE $X +- 1.96 \frac \sigma {\sqrt{n}}$**


```python
#In: 
# Construindo um IC
(mean - 1.96 * s_over_n, mean + 1.96 * s_over_n)
```




    (14.705194475423134, 15.814805524576865)



**95% de chance da média cair no intervalo de tamanho n acima. O mesmo não inclui o 22, então podemos assumir que o valor é não esperado.**

Observe que existe uma chance de cometermos erros, qual é?

## A situação mais comum na vida real

Normalmente temos *uma amostra* da população apenas. Daí não conhecemos a distribuição amostral. Mas gostaríamos de a partir da nossa amostra estimar onde está a estatística para a população. 

Exemplo: queremos estimar qual a proporção de pessoas que gostará do produto (a estatística) entre todos os usuários (a população) a partir do cálculo da proporção de pessoas que gostou do produto (a mesma estatística) em um teste com 100 pessoas (a amostra).

Repare que se conhecermos como a estatística varia na distribuição amostral (ex: 2 pontos pra mais ou pra menos cobrem 99% dos casos) e temos a estatística calculada para a amostra, poderíamos estimar uma faixa de valores onde achamos que a estatística está para a população _com 99% de confiança_.

### A ideia central que usaremos

Para exemplificar o caso acima, vamos explorar alguns dados reais de salários da Billboard. 
A ideia principal que usaremos, em uma técnica chamada *boostrapping* é que _usar a amostra como substituto da população e simular a amostragem através de reamostragem com reposição fornece uma estimativa precisa da variação na distribuição amostral_. 

Para implementar o Bootstrap, vamos implementar uma função para o bootstrap_raw. A mesma faz uso da função `df.sample` que gera uma amostra aleatória de n elementos retirados do df. O funcionamento é similar a função `np.random.choice`. Note que estamos fazendo um bootstrap da mediana, podemos fazer patra outras medidas centrais.

1. Dado `n` e `size`
2. Gere `n` amostras de tamanho `size` com reposição
3. Tira a mediana (podia ser média ou qualquer outra medida central)
4. Retorne as novas amostras e veja a distribuição das mesmas


```python
#In: 
def bootstrap_median(df, col, n=5000, size=None):
    if size is None:
        size = len(df)
    values = np.zeros(n)
    for i in range(n):
        sample = df.sample(size, replace=True)
        values[i] = sample[col].median()
    return values
```


```python
#In: 
# 1. lendo dados
df = pd.read_csv('https://media.githubusercontent.com/media/icd-ufmg/material/master/aulas/09-ICs/billboard_2000_2018_spotify_lyrics.csv',
                 encoding='iso-8859-1', na_values='unknown')
# 2. removendo na
df = df.dropna()
df = df[['title', 'main_artist', 'duration_ms']]

# 3. convertendo para minutos
df['duration_m'] = df['duration_ms'] / (60*1000)

# 4. apagando coluna antiga
del df['duration_ms']
df.head(5)
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
      <th>title</th>
      <th>main_artist</th>
      <th>duration_m</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1017</th>
      <td>Unsteady</td>
      <td>x ambassadors</td>
      <td>3.225767</td>
    </tr>
    <tr>
      <th>1018</th>
      <td>Too Much Sauce</td>
      <td>dj esco</td>
      <td>3.718217</td>
    </tr>
    <tr>
      <th>1021</th>
      <td>Key To The Streets</td>
      <td>yfn lucci</td>
      <td>4.699783</td>
    </tr>
    <tr>
      <th>1024</th>
      <td>Careless Whisper</td>
      <td>wham!</td>
      <td>5.205917</td>
    </tr>
    <tr>
      <th>1025</th>
      <td>Faith</td>
      <td>george michael</td>
      <td>3.220000</td>
    </tr>
  </tbody>
</table>
</div>



Imagine por agora que os dados que temos de apenas 100 música Billboard são completos. Sei que tenho mais no `df`, mas preciso de small data aqui para executar o notebook.


```python
#In: 
df = df.sample(100)
```


```python
#In: 
plt.hist(df['duration_m'], bins=30, edgecolor='k')
plt.xlabel('Duração em minutos')
plt.ylabel('P[mediana]')
despine()
```


    
![png](09-ics_files/09-ics_65_0.png)
    


A mediana foi de:


```python
#In: 
df['duration_m'].median()
```




    3.7401083333333336



Se calcularmos a mediana do números de novas músicas para três amostras de 1000 elementos, teremos 3 resultados diferentes. Estamos usando 1000 pois é o mesmo tamanho da nossa **falsa** população. A ideia do bootstrap é usar amostras da amostra original como diferentes visões da população.


```python
#In: 
for _ in range(3):
    print(df['duration_m'].sample(len(df), replace=True).median())
    print()
```

    3.7260833333333334
    
    3.7479
    
    3.7006666666666668
    


Se fizermos isso muitas vezes podemos ver como essa variação acontece. Em particular, vamos fazer 10000 vezes. Note que o código abaixo é essenciamente o mesmo da função `bootstrap` acima.


```python
#In: 
S = len(df)
N = 5000
values = np.zeros(N)
for i in range(N):
    sample = df.sample(S, replace=True)
    values[i] = sample['duration_m'].median()
print(values)
```

    [3.66665833 3.73621667 3.76645    ... 3.72608333 3.796      3.80666667]



```python
#In: 
plt.hist(values, bins=30, edgecolor='k')
plt.xlabel('Mediana da Amostra de Tamanho 1000')
plt.ylabel('P(mediana)')
despine()
```


    
![png](09-ics_files/09-ics_72_0.png)
    


Usando a função score at percentile sabemos onde ficam 95% dos dados sintéticos.


```python
#In: 
np.percentile(values, 2.5)
```




    3.6464499999999997




```python
#In: 
np.percentile(values, 97.5)
```




    3.8417858333333332



Acamos de construir um **IC**.

Pegando por partes: 

 * Consideramos a amostra $A$ que tem tamanho $n$ como sendo um substituto da população  
 * Repetimos $b$ vezes o seguinte processo: criamos uma amostra de tamanho proporcional a $n$ obtendo elementos aleatoriamente de $A$, repondo cada elemento depois de cada sorteio. 
 * Calculamos a estatística $e$ que nos interessa (média, mediana, desvio padrão, o que for) para cada uma das $b$ amostras. 
 
Como resultado, sabemos como a estatística $e$ varia em uma simulação de $b$ amostragens. Podemos usar os percentis para criar um IC. Assim, se estimamos que em $P[E <= e_l] = 0.025$ e $P[E > e_h] = 0.025$ ou $P[E <= e_h] = 0.975$, nosso IC será: $(e - e_l, e + e_h)$.

1. Podemos usar bootstrap para tendências centrais não extremas.
1. O bootstrap falha quando os dados tem cauda pesada.

Um pouco de código de animações abaixo, pode ignorar o mesmo!

**Ingore daqui, caso queira**


```python
#In: 
def update_hist(num, data):
    plt.cla()
    plt.hist(data[0:100 * (num+1)], bins=20, edgecolor='k')
    plt.xlabel('Mediana da Amostra de Tamanho 1000')
    plt.ylabel('P(mediana)')
    despine()
```


```python
#In: 
values = bootstrap_median(df, 'duration_m')
fig = plt.figure()
ani = animation.FuncAnimation(fig, update_hist, 30, fargs=(values, ))
HTML(ani.to_html5_video())
```




<video width="1600" height="1000" controls autoplay loop>
  <source type="video/mp4" src="data:video/mp4;base64,AAAAIGZ0eXBNNFYgAAACAE00ViBpc29taXNvMmF2YzEAAAAIZnJlZQABLrJtZGF0AAACrgYF//+q
3EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NCByMzA5NSBiYWVlNDAwIC0gSC4yNjQvTVBF
Ry00IEFWQyBjb2RlYyAtIENvcHlsZWZ0IDIwMDMtMjAyMiAtIGh0dHA6Ly93d3cudmlkZW9sYW4u
b3JnL3gyNjQuaHRtbCAtIG9wdGlvbnM6IGNhYmFjPTEgcmVmPTMgZGVibG9jaz0xOjA6MCBhbmFs
eXNlPTB4MzoweDExMyBtZT1oZXggc3VibWU9NyBwc3k9MSBwc3lfcmQ9MS4wMDowLjAwIG1peGVk
X3JlZj0xIG1lX3JhbmdlPTE2IGNocm9tYV9tZT0xIHRyZWxsaXM9MSA4eDhkY3Q9MSBjcW09MCBk
ZWFkem9uZT0yMSwxMSBmYXN0X3Bza2lwPTEgY2hyb21hX3FwX29mZnNldD0tMiB0aHJlYWRzPTEy
IGxvb2thaGVhZF90aHJlYWRzPTIgc2xpY2VkX3RocmVhZHM9MCBucj0wIGRlY2ltYXRlPTEgaW50
ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MyBi
X3B5cmFtaWQ9MiBiX2FkYXB0PTEgYl9iaWFzPTAgZGlyZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29w
PTAgd2VpZ2h0cD0yIGtleWludD0yNTAga2V5aW50X21pbj01IHNjZW5lY3V0PTQwIGludHJhX3Jl
ZnJlc2g9MCByY19sb29rYWhlYWQ9NDAgcmM9Y3JmIG1idHJlZT0xIGNyZj0yMy4wIHFjb21wPTAu
NjAgcXBtaW49MCBxcG1heD02OSBxcHN0ZXA9NCBpcF9yYXRpbz0xLjQwIGFxPTE6MS4wMACAAAA/
hWWIhAA///73aJ8Cm1pDeoDklcUl20+B/6tncHzIZiN/OAAAAwAAAwAAAwAAAwAZdOpYO8Tofu/8
QAAAAwAABEAAABowAAFGAAAW8AAB9AAANRwnzoJIVICfpy57i64hDOgyyNJZ5ovLkJbtOksOMAHy
ngSwAADYBNs+g88zN0X4kacZYtlml5tEZ1KbLiHiGxYF2jE8GZRVf+Riz/y/sUGiYLbtJ+XRLuD8
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
MA==
">
  Your browser does not support the video tag.
</video>




    
![png](09-ics_files/09-ics_79_1.png)
    


**Até aqui!**

## Brincando com o bootstrap

E se a amostra fosse muito menor? Parece que temos alguns problemas! Voltamos a estimar o histograma, qual o motivo?


```python
#In: 
S = 2
N = 5000
values = bootstrap_median(df, 'duration_m', size=S, n=N)
plt.hist(values, bins=20, edgecolor='k');
plt.xlabel('Mediana da Amostra de Tamanho 2')
plt.ylabel('P(mediana)')
despine()
```


    
![png](09-ics_files/09-ics_82_0.png)
    


No geral, devemos gerar amostras perto do tamanho da amostra original, ou pelo menos algum valor alto. Note que algumas centenas, neste caso, já se comporta similar ao caso com 1000.


```python
#In: 
S = 30
N = 5000
values = bootstrap_median(df, 'duration_m', size=S, n=N)
plt.hist(values, bins=20, edgecolor='k');
plt.xlabel('Mediana da Amostra de Tamanho 30')
plt.ylabel('P(mediana)')
despine()
```


    
![png](09-ics_files/09-ics_84_0.png)
    


Um novo problema. Se forem poucas amostras, mesmo sendo um `S` razoável?!


```python
#In: 
S = len(df)
N = 10
values = bootstrap_median(df, 'duration_m', size=S, n=N)
plt.hist(values, bins=10, edgecolor='k');
plt.xlabel('Mediana de 10 amstras de Tamanho 100')
plt.ylabel('P(mediana)')
despine()
```


    
![png](09-ics_files/09-ics_86_0.png)
    


Gerando 2000 bootstraps e observar os que não capturam a mediana. O código abaixo demora usar um pouco de mágia numpy para executar de forma rápida.

## Comparando o Bootstrap com o caso Teórico


```python
#In: 
# voltando para as moedas

TAMANHO_AMOSTRA = 100
resultados = []
for i in range(TAMANHO_AMOSTRA):
    jogadas = np.random.randint(0, 2, size=30) # joga 30 moedas para cima
    n_caras = (jogadas == 1).sum()             # conta quantas foram == 1, caras
    resultados.append(n_caras)
resultados = np.array(resultados)
```


```python
#In: 
def bootstrap_mean(x, n=5000, size=None):
    if size is None:
        size = len(x)
    values = np.zeros(n)
    for i in range(n):
        sample = np.random.choice(x, size=size, replace=True)
        values[i] = sample.mean()
    return values
```


```python
#In: 
boot_samples = bootstrap_mean(resultados)
```


```python
#In: 
np.percentile(boot_samples, 2.5)
```




    13.88




```python
#In: 
np.percentile(boot_samples, 97.5)
```




    14.89




```python
#In: 
s = np.std(resultados, ddof=1)
s_over_n = s / np.sqrt(len(resultados))
mean = np.mean(resultados)
```


```python
#In: 
(mean - 1.96 * s_over_n, mean + 1.96 * s_over_n)
```




    (13.887859806287713, 14.892140193712288)




```python
#In: 
plt.hist(boot_samples, edgecolor='k');
```


    
![png](09-ics_files/09-ics_96_0.png)
    



```python
#In: 

```
