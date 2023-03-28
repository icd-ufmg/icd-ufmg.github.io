---
layout: page
title: Tabelas e Tipos de Dados
nav_order: 2
---

[<img src="./colab_favicon_small.png" style="float: right;">](https://colab.research.google.com/github/icd-ufmg/icd-ufmg.github.io/blob/master/_lessons/02-tabelas.ipynb)


# Tabelas e Tipos de Dados

{: .no_toc .mb-2 }

Aprendendo como criar e manipular tabelas.
{: .fs-6 .fw-300 }

{: .no_toc .text-delta }
Resultados Esperados

1. Aprender o básico de Pandas
1. Entender diferentes tipos de dados
1. Básico de filtros e seleções
1. Aplicação de filtros básicos para gerar insights nos dados de dados tabulares



---
**Sumário**
1. TOC
{:toc}
---

## Introdução

Neste notebook vamos explorar um pouco de dados tabulares, ou seja, tabelas. Tabelas de dados representam um dos tipos de dados mais comuns para o cientista de dados. Pense nas inúmeras tabelas Excell que você já trabalhou com.

A principal biblioteca para leitura de dados tabulares em Python se chama **pandas**. A mesma é bastante poderosa implementando uma série de operações de bancos de dados (e.g., groupby e join). Nossa discussão será focada em algumas das funções principais do pandas que vamos explorar no curso. Existe uma série ampla de funcionalidades que a biblioteca (além de outras) vai trazer. 

Caso necessite de algo além da aula, busque na documentação da biblioteca. Por fim, durante esta aula, também vamos aprender um pouco de bash.

### Imports básicos

A maioria dos nossos notebooks vai iniciar com os imports abaixo.
1. pandas: dados tabulates
1. matplotlib: gráficos e plots

A chamada `plt.ion` habilita gráficos do matplotlib no notebook diretamente. Caso necesse salvar alguma figura, chame `plt.savefig` após seu plot.


```python
#In: 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
```

<details>
<summary>Código de Configurar Plot (Oculto)</summary>


```python
#In: 
plt.rcParams['figure.figsize'] = (16, 10)

plt.rcParams['axes.axisbelow'] = True 
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.ymargin'] = 0.1

plt.rcParams['font.family'] = 'serif'

plt.rcParams['grid.color'] = 'lightgrey'
plt.rcParams['grid.linewidth'] = .1

plt.rcParams['xtick.bottom'] = True
plt.rcParams['xtick.direction'] = 'out' 
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['xtick.major.size'] = 12
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.size'] = 6
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['xtick.minor.visible'] = True

plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.major.size'] = 12
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.size'] = 6
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['ytick.minor.visible'] = True

plt.rcParams['legend.fontsize'] = 16

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['lines.markersize'] = 80
```


```python
#In: 
plt.style.use('tableau-colorblind10')
plt.ion();
```

</details>

## Series

Existem dois tipos base de dados em pandas. O primeiro, Series, representa uma coluna de dados. Um combinação de Series vira um DataFrame (mais abaixo). Diferente de um vetor `numpy`, a Series de panda captura uma coluna de dados (ou vetor) indexado. Isto é, podemos nomear cada um dos valores.


```python
#In: 
data = pd.Series(
    [0.25, 0.5, 0.75, 1.0],
    index=['a', 'b', 'c', 'd']
)
```


```python
#In: 
data
```




    a    0.25
    b    0.50
    c    0.75
    d    1.00
    dtype: float64



Note que podemos usar como um vetor


```python
#In: 
data[0]
```




    0.25



Porém o índice nos ajuda. Para um exemplo trivial como este não será tão interessante, mas vamos usar o mesmo.


```python
#In: 
data.index
```




    Index(['a', 'b', 'c', 'd'], dtype='object')



Com .loc acessamos uma linha do índice com base no nome. Então:

1. `series.loc[objeto_python]` - valor com o devido nome.
1. `series.iloc[int]` - i-ésimo elemento da Series.


```python
#In: 
data.loc['a']
```




    0.25




```python
#In: 
data.loc['b']
```




    0.5



Com `iloc` acessamos por número da linha, estilho um vetor.


```python
#In: 
data.iloc[0]
```




    0.25




```python
#In: 
data[0]
```




    0.25



## Data Frames

Ao combinar várias Series com um índice comum, criamos um **DataFrame**. Não é tão comum gerar os mesmos na mão como estamos fazendo, geralmente carregamos DataFrames de arquivos `.csv`, `.json` ou até de sistemas de bancos de dados `mariadb`. De qualquer forma, use os exemplos abaixo para entender a estrutura de um dataframe.

Lembre-se que {}/dict é um dicionário (ou mapa) em Python. Podemos criar uma série a partir de um dicionário
index->value


```python
#In: 
area_dict = {'California': 423967,
             'Texas': 695662,
             'New York': 141297,
             'Florida': 170312,
             'Illinois': 149995}
```

A linha abaixo pega todas as chaves.


```python
#In: 
list(area_dict.keys())
```




    ['California', 'Texas', 'New York', 'Florida', 'Illinois']



Agora todas as colunas


```python
#In: 
list(area_dict.values())
```




    [423967, 695662, 141297, 170312, 149995]



Acessando um valor.


```python
#In: 
area_dict['California']
```




    423967



Podemos criar a série a partir do dicionário, cada chave vira um elemento do índice. Os valores viram os dados do vetor.


```python
#In: 
area = pd.Series(area_dict)
area
```




    California    423967
    Texas         695662
    New York      141297
    Florida       170312
    Illinois      149995
    dtype: int64



Agora, vamos criar outro dicionário com a população dos estados.


```python
#In: 
pop_dict = {'California': 38332521,
            'Texas': 26448193,
            'New York': 19651127,
            'Florida': 19552860,
            'Illinois': 12882135}
pop = pd.Series(pop_dict)
pop
```




    California    38332521
    Texas         26448193
    New York      19651127
    Florida       19552860
    Illinois      12882135
    dtype: int64



Por fim, observe que o DataFrame é uma combinação de Series. Cada uma das Series vira uma coluna da tabela de dados.


```python
#In: 
data = pd.DataFrame({'area':area, 'pop':pop})
data
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
      <th>area</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>423967</td>
      <td>38332521</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>695662</td>
      <td>26448193</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>141297</td>
      <td>19651127</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>170312</td>
      <td>19552860</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>149995</td>
      <td>12882135</td>
    </tr>
  </tbody>
</table>
</div>



Agora o use de `.loc e .iloc` deve ficar mais claro, observe os exemplos abaixo.


```python
#In: 
data.loc['California']
```




    area      423967
    pop     38332521
    Name: California, dtype: int64




```python
#In: 
data.loc[['California', 'Texas']]
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
      <th>area</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>423967</td>
      <td>38332521</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>695662</td>
      <td>26448193</td>
    </tr>
  </tbody>
</table>
</div>



Note que o uso de `iloc` retorna a i-ésima linha. O problema é que nem sempre nos dataframes esta ordem vai fazer sentido. O `iloc` acaba sendo mais interessante para iteração (e.g., passar por todas as linhas.)


```python
#In: 
data.iloc[0]
```




    area      423967
    pop     38332521
    Name: California, dtype: int64



## Slicing

Agora, podemos realizar slicing no DataFrame. Slicing é uma operação Python que retorna sub-listas/sub-vetores. Caso não conheça, tente executar o exemplo abaixo:

```python
#In: 
l = []
l = [7, 1, 3, 5, 9]
print(l[0])
print(l[1])
print(l[2])

# Agora, l[bg:ed] retorna uma sublista iniciando em bg e terminando em ed-1
print(l[1:4])
```


```python
#In: 
l = []
l = [7, 1, 3, 5, 9]
print(l[0])
print(l[1])
print(l[2])

# Agora, l[bg:ed] retorna uma sublista iniciando em bg e terminando em ed-1
print(l[1:4])
```

    7
    1
    3
    [1, 3, 5]


Voltando para o nosso **dataframe**, podemos realizar o slicing usando o `.iloc`.


```python
#In: 
data.iloc[2:4]
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
      <th>area</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>New York</th>
      <td>141297</td>
      <td>19651127</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>170312</td>
      <td>19552860</td>
    </tr>
  </tbody>
</table>
</div>



## Modificando DataFrames

Series e DataFrames são objetos mutáveis em Python. Podemos adicionar novas colunas em DataFrama facilmente da mesma forma que adicionamos novos valores em um mapa. Por fim, podemos também mudar o valor de linhas específicas e adicionar novas linhas.


```python
#In: 
data['density'] = data['pop'] / data['area']
data.loc['Texas']
```




    area       6.956620e+05
    pop        2.644819e+07
    density    3.801874e+01
    Name: Texas, dtype: float64




```python
#In: 
df = data
```


```python
#In: 
df.index
```




    Index(['California', 'Texas', 'New York', 'Florida', 'Illinois'], dtype='object')



## Arquivos

Antes de explorar DataFrames em arquivos, vamos ver como um notebook na é um shell bastante poderoso. Ao usar uma exclamação (!) no notebook Jupyter, conseguimos executar comandos do shell do sistema. Em particular, aqui estamos executando o comando ls para indentificar os dados da pasta atual.

Tudo que executamos com `!` é um comando do terminal do unix. Então, este notebook só deve executar as linhas abaixo em um `Mac` ou `Linux`.


```python
#In: 
!ls .
```

    01-causalidade.md     15-linear
    02-tabelas	      15-linear_files
    02-tabelas.ipynb      15-linear.ipynb
    02-tabelas.md	      15-linear.md
    03-viz		      16-vero
    03-viz_files	      16-vero_files
    03-viz.ipynb	      16-vero.ipynb
    03-viz.md	      16-vero.md
    04-stat		      17-gradiente
    04-stat_files	      17-gradiente_files
    04-stat.ipynb	      17-gradiente.ipynb
    04-stat.md	      17-gradiente.md
    05-prob		      18-multipla
    05-prob_files	      18-multipla_files
    05-prob.ipynb	      18-multipla.ipynb
    05-prob.md	      18-multipla.md
    06-risco	      19-logistica
    06-risco_files	      19-logistica_files
    06-risco.ipynb	      19-logistica.ipynb
    06-risco.md	      19-logistica.md
    07-tcl		      20-knn
    07-tcl_files	      20-knn_files
    07-tcl.ipynb	      20-knn.ipynb
    07-tcl.md	      20-knn.md
    08-amostragem	      21-pratica
    08-amostragem_files   21-pratica_files
    08-amostragem.ipynb   21-pratica.ipynb
    08-amostragem.md      21-pratica.md
    09-ics		      22-svd-e-pca
    09-ics_files	      22-svd-e-pca_files
    09-ics.ipynb	      22-svd-e-pca.ipynb
    09-ics.md	      22-svd-e-pca.md
    10-ab		      26b-tutorial-sklearn-classification.ipynb
    10-ab_files	      26-tutorial-sklearn-regressao.ipynb
    10-ab.ipynb	      27-revisao
    10-ab.md	      27-revisao_files
    11-hipoteses	      27-revisao.ipynb
    11-hipoteses_files    27-revisao.md
    11-hipoteses.ipynb    baby.csv
    11-hipoteses.md       capital.json
    12-causalidade	      colab_favicon.ico
    12-causalidade_files  colab_favicon.png
    12-causalidade.ipynb  colab_favicon_small.png
    12-causalidade.md     compile_all.sh
    13-poder	      compile_notebook.sh
    13-poder_files	      dom-casmurro.txt
    13-poder.ipynb	      drum.wav
    13-poder.md	      f2.png
    14-correlacao	      f.png
    14-correlacao_files   proglanguages.json
    14-correlacao.ipynb   states.txt
    14-correlacao.md      walmart.csv


Com a opção -lha, mostramos meta-dados dos arquivos como o owner, tamanho e permissões. Note que todos os arquivos são .csv, isto é comma separated.


```python
#In: 
! ls -lha .
```

    total 180M
    drwxr-xr-x 46 flaviovdf flaviovdf 4.0K Mar 21 18:36 .
    drwxr-xr-x  9 flaviovdf flaviovdf 4.0K Mar 14 18:52 ..
    -rw-r--r--  1 flaviovdf flaviovdf  618 Mar 14 18:20 01-causalidade.md
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Mar 21 18:19 02-tabelas
    -rw-r--r--  1 flaviovdf flaviovdf 107K Mar 21 18:36 02-tabelas.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  49K Mar 21 18:16 02-tabelas.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Mar 21 18:36 03-viz
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Mar 21 18:36 03-viz_files
    -rw-r--r--  1 flaviovdf flaviovdf 1.2M Mar 21 18:36 03-viz.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  40K Mar 21 18:36 03-viz.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Mar 14 18:20 04-stat
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Mar 14 18:20 04-stat_files
    -rw-r--r--  1 flaviovdf flaviovdf 360K Mar 14 18:20 04-stat.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  26K Mar 14 18:20 04-stat.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Mar 14 18:20 05-prob
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Mar 14 18:20 05-prob_files
    -rw-r--r--  1 flaviovdf flaviovdf 892K Mar 14 18:20 05-prob.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  35K Mar 14 18:20 05-prob.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Mar 14 18:20 06-risco
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Mar 14 18:20 06-risco_files
    -rw-r--r--  1 flaviovdf flaviovdf 291K Mar 14 18:20 06-risco.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  15K Mar 14 18:20 06-risco.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Mar 14 18:20 07-tcl
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Mar 14 18:20 07-tcl_files
    -rw-r--r--  1 flaviovdf flaviovdf 445K Mar 14 18:20 07-tcl.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  14K Mar 14 18:20 07-tcl.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Mar 14 18:20 08-amostragem
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Mar 14 18:20 08-amostragem_files
    -rw-r--r--  1 flaviovdf flaviovdf 232K Mar 14 18:20 08-amostragem.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf 117K Mar 14 18:20 08-amostragem.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Mar 14 18:20 09-ics
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Mar 14 18:20 09-ics_files
    -rw-r--r--  1 flaviovdf flaviovdf 692K Mar 14 18:20 09-ics.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  91K Mar 14 18:20 09-ics.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Mar 14 18:20 10-ab
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Mar 14 18:20 10-ab_files
    -rw-r--r--  1 flaviovdf flaviovdf 172K Mar 14 18:20 10-ab.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  17K Mar 14 18:20 10-ab.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Mar 14 18:20 11-hipoteses
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Mar 14 18:20 11-hipoteses_files
    -rw-r--r--  1 flaviovdf flaviovdf 301K Mar 14 18:20 11-hipoteses.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  94K Mar 14 18:20 11-hipoteses.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Mar 14 18:20 12-causalidade
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Mar 14 18:20 12-causalidade_files
    -rw-r--r--  1 flaviovdf flaviovdf  52K Mar 14 18:20 12-causalidade.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf 9.6K Mar 14 18:20 12-causalidade.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Mar 14 18:20 13-poder
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Mar 14 18:20 13-poder_files
    -rw-r--r--  1 flaviovdf flaviovdf 837K Mar 14 18:20 13-poder.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  25K Mar 14 18:20 13-poder.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Mar 14 18:20 14-correlacao
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Mar 14 18:20 14-correlacao_files
    -rw-r--r--  1 flaviovdf flaviovdf 1.8M Mar 14 18:20 14-correlacao.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  39K Mar 14 18:20 14-correlacao.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Mar 14 18:20 15-linear
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Mar 14 18:20 15-linear_files
    -rw-r--r--  1 flaviovdf flaviovdf 992K Mar 14 18:20 15-linear.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  25K Mar 14 18:20 15-linear.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Mar 14 18:20 16-vero
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Mar 14 18:20 16-vero_files
    -rw-r--r--  1 flaviovdf flaviovdf 1.0M Mar 14 18:20 16-vero.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  30K Mar 14 18:20 16-vero.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Mar 14 18:20 17-gradiente
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Mar 14 18:20 17-gradiente_files
    -rw-r--r--  1 flaviovdf flaviovdf 731K Mar 14 18:20 17-gradiente.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  58K Mar 14 18:20 17-gradiente.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Mar 14 18:20 18-multipla
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Mar 14 18:20 18-multipla_files
    -rw-r--r--  1 flaviovdf flaviovdf 671K Mar 14 18:20 18-multipla.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf 136K Mar 14 18:20 18-multipla.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Mar 14 18:20 19-logistica
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Mar 14 18:20 19-logistica_files
    -rw-r--r--  1 flaviovdf flaviovdf 886K Mar 14 18:20 19-logistica.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  79K Mar 14 18:20 19-logistica.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Mar 14 18:20 20-knn
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Mar 14 18:20 20-knn_files
    -rw-r--r--  1 flaviovdf flaviovdf 790K Mar 14 18:20 20-knn.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  19K Mar 14 18:20 20-knn.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Mar 14 18:20 21-pratica
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Mar 14 18:20 21-pratica_files
    -rw-r--r--  1 flaviovdf flaviovdf 280K Mar 14 18:20 21-pratica.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  72K Mar 14 18:20 21-pratica.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Mar 14 18:20 22-svd-e-pca
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Mar 14 18:20 22-svd-e-pca_files
    -rw-r--r--  1 flaviovdf flaviovdf 9.5M Mar 14 18:20 22-svd-e-pca.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf 6.7M Mar 14 18:20 22-svd-e-pca.md
    -rw-r--r--  1 flaviovdf flaviovdf  47K Mar 14 18:20 26b-tutorial-sklearn-classification.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf 203K Mar 14 18:20 26-tutorial-sklearn-regressao.ipynb
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Mar 14 18:20 27-revisao
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Mar 14 18:20 27-revisao_files
    -rw-r--r--  1 flaviovdf flaviovdf  63K Mar 14 18:20 27-revisao.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  17K Mar 14 18:20 27-revisao.md
    -rw-r--r--  1 flaviovdf flaviovdf 148M Mar 21 18:02 baby.csv
    -rw-r--r--  1 flaviovdf flaviovdf 172K Mar 14 18:20 capital.json
    -rw-r--r--  1 flaviovdf flaviovdf  72K Mar 14 18:20 colab_favicon.ico
    -rw-r--r--  1 flaviovdf flaviovdf 7.8K Mar 14 18:20 colab_favicon.png
    -rw-r--r--  1 flaviovdf flaviovdf 2.2K Mar 14 18:20 colab_favicon_small.png
    -rw-r--r--  1 flaviovdf flaviovdf   85 Mar 14 18:20 compile_all.sh
    -rw-r--r--  1 flaviovdf flaviovdf 1.1K Mar 14 18:20 compile_notebook.sh
    -rw-r--r--  1 flaviovdf flaviovdf 401K Mar 14 18:20 dom-casmurro.txt
    -rw-r--r--  1 flaviovdf flaviovdf 331K Mar 14 18:20 drum.wav
    -rw-r--r--  1 flaviovdf flaviovdf 125K Mar 14 18:20 f2.png
    -rw-r--r--  1 flaviovdf flaviovdf  57K Mar 14 18:20 f.png
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Mar 14 18:20 .ipynb_checkpoints
    -rw-r--r--  1 flaviovdf flaviovdf  29K Mar 14 18:20 .nbgrader.log
    -rw-r--r--  1 flaviovdf flaviovdf 3.1K Mar 14 18:20 proglanguages.json
    -rw-r--r--  1 flaviovdf flaviovdf 131K Mar 14 18:20 states.txt
    -rw-r--r--  1 flaviovdf flaviovdf 715K Mar 14 18:20 walmart.csv


Vamos identificar qual a cara de um csv. O programa `head` imprime as primeiras `n` linhas de um arquivo.


```python
#In: 
! head baby.csv
```

    Id,Name,Year,Gender,State,Count
    1,Mary,1910,F,AK,14
    2,Annie,1910,F,AK,12
    3,Anna,1910,F,AK,10
    4,Margaret,1910,F,AK,8
    5,Helen,1910,F,AK,7
    6,Elsie,1910,F,AK,6
    7,Lucy,1910,F,AK,6
    8,Dorothy,1910,F,AK,5
    9,Mary,1911,F,AK,12


Observe como o comando `head` nos ajuda a entender o arquivo `.csv`. Sabemos quais colunas e qual o separador do mesmo.

## Baby Names

É bem mais comum fazer uso de DataFrames que já existem em arquivos. Note que o trabalho do cientista de dados nem sempre vai ter tais arquivos prontos. Em várias ocasiões, você vai ter que coletar e organizar os mesmos. Limpeza e coleta de dados é uma parte fundamental do seu trabalho. Durante a matéria, boa parte dos notebooks já vão ter dados prontos.


```python
#In: 
df = pd.read_csv('https://media.githubusercontent.com/media/icd-ufmg/material/master/aulas/03-Tabelas-e-Tipos-de-Dados/baby.csv')
df = df.drop('Id', axis='columns') # remove a coluna id, serve de nada
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
      <th>Name</th>
      <th>Year</th>
      <th>Gender</th>
      <th>State</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mary</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Annie</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Anna</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Margaret</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Helen</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>7</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5647421</th>
      <td>Seth</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647422</th>
      <td>Spencer</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647423</th>
      <td>Tyce</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647424</th>
      <td>Victor</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647425</th>
      <td>Waylon</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>5647426 rows × 5 columns</p>
</div>



O método `head` do notebook retorna as primeiras `n` linhas do mesmo. Use tal método para entender seus dados. **Sempre olhe para seus dados.** Note como as linhas abaixo usa o `loc` e `iloc` para entender um pouco a estrutura dos mesmos.


```python
#In: 
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
      <th>Name</th>
      <th>Year</th>
      <th>Gender</th>
      <th>State</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mary</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Annie</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Anna</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Margaret</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Helen</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



O método `head` do notebook retorna as primeiras `n` linhas do mesmo. Use tal método para entender seus dados. **Sempre olhe para seus dados.** Note como as linhas abaixo usa o `loc` e `iloc` para entender um pouco a estrutura dos mesmos.


```python
#In: 
df.head(6)
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
      <th>Name</th>
      <th>Year</th>
      <th>Gender</th>
      <th>State</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mary</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Annie</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Anna</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Margaret</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Helen</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Elsie</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
#In: 
df[10:15]
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
      <th>Name</th>
      <th>Year</th>
      <th>Gender</th>
      <th>State</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>Ruth</td>
      <td>1911</td>
      <td>F</td>
      <td>AK</td>
      <td>7</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Annie</td>
      <td>1911</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Elizabeth</td>
      <td>1911</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Helen</td>
      <td>1911</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Mary</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
#In: 
df.iloc[0:6]
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
      <th>Name</th>
      <th>Year</th>
      <th>Gender</th>
      <th>State</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mary</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Annie</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Anna</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Margaret</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Helen</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Elsie</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
#In: 
df[['Name', 'Gender']].head(6)
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
      <th>Name</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mary</td>
      <td>F</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Annie</td>
      <td>F</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Anna</td>
      <td>F</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Margaret</td>
      <td>F</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Helen</td>
      <td>F</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Elsie</td>
      <td>F</td>
    </tr>
  </tbody>
</table>
</div>




```python
#In: 
df[['Name', 'Gender']].head(6)
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
      <th>Name</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mary</td>
      <td>F</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Annie</td>
      <td>F</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Anna</td>
      <td>F</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Margaret</td>
      <td>F</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Helen</td>
      <td>F</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Elsie</td>
      <td>F</td>
    </tr>
  </tbody>
</table>
</div>



## Groupby

Vamos responder algumas perguntas com a função groupby. Lembrando a ideia é separar os dados com base em valores comuns, ou seja, agrupar por nomes e realizar alguma operação. O comando abaixo agrupa todos os recem-náscidos por nome. Imagine a mesma fazendo uma operação equivalente ao laço abaixo:

```python
#In: 
buckets = {}                    # Mapa de dados
names = set(df['Name'])         # Conjunto de nomes únicos
for idx, row in df.iterrows():  # Para cada linha dos dados
    name = row['Name']
    if name not in buckets:
        buckets[name] = []      # Uma lista para cada nome
    buckets[name].append(row)   # Separa a linha para cada nome
```

O código acima é bastante lento!!! O groupby é optimizado. Com base na linha abaixo, o mesmo nem retorna nehum resultado ainda. Apenas um objeto onde podemos fazer agregações.


```python
#In: 
gb = df.groupby('Name')
type(gb)
```




    pandas.core.groupby.generic.DataFrameGroupBy



Agora posso agregar todos os nomes com alguma operação. Por exemplo, posso somar a quantidade de vezes que cada nome ocorre. Em python, seria o seguinte código.

```python
#In: 
sum_ = {}                       # Mapa de dados
for name in buckets:            # Para cada nomee
    sum_[name] = 0
    for row in buckets[name]:   # Para cada linha com aquele nome, aggregate (some)
        sum_[name] += row['Count']
```

Observe o resultado da agregação abaixo. Qual o problema com a coluna `Year`??


```python
#In: 
gb[['Year', 'Count']].mean()
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
      <th>Year</th>
      <th>Count</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Aaban</th>
      <td>2013.500000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>Aadan</th>
      <td>2009.750000</td>
      <td>5.750000</td>
    </tr>
    <tr>
      <th>Aadarsh</th>
      <td>2009.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Aaden</th>
      <td>2010.015306</td>
      <td>17.479592</td>
    </tr>
    <tr>
      <th>Aadhav</th>
      <td>2014.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Zyrah</th>
      <td>2012.000000</td>
      <td>5.500000</td>
    </tr>
    <tr>
      <th>Zyren</th>
      <td>2013.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>Zyria</th>
      <td>2006.714286</td>
      <td>5.785714</td>
    </tr>
    <tr>
      <th>Zyriah</th>
      <td>2009.666667</td>
      <td>6.444444</td>
    </tr>
    <tr>
      <th>Zyshonne</th>
      <td>1998.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
<p>30274 rows × 2 columns</p>
</div>



Não faz tanto sentido somar o ano, embora seja um número aqui representa uma categoria. Vamos somar as contagens apenas.

**Observe como a chamada abaixo tem um enter depois de cada ponto. Isso é um truque para deixar o código mais legível**

```python
#In: 

# Leia dessa forma

(dados[['coluna1', 'coluna2']].    # selecione algumas colunas
 operacao_a().                     # realize uma operação
 operacao_b()                      # realize outra
)

# É o mesmo de

dados[['coluna1', 'coluna2']].operacao_a().operacao_b()

```

Cada chamada pandas retorna um DataFrame novo. Ao fazer `.` chamamos um novo método. Quebrando a linha no ponto fica mais fácil separa a lógica de cada operação. Compare os casos abaixo.

```python
#In: 

# Leia dessa forma

(dados[['coluna1', 'coluna2']].    # selecione algumas colunas
 operacao_a().                     # realize uma operação
 operacao_b().                     # realize outra
 operaca_c().
 operaca_d().
 operaca_e().
 operaca_f().
)

# É o mesmo de

dados[['coluna1', 'coluna2']].operacao_a().operacao_c().operacao_d().operacao_e().operacao_f()

```


```python
#In: 
(gb['Count'].
 sum().
 sort_values()
)
```




    Name
    Zyshonne          5
    Makenlee          5
    Makenlie          5
    Makinlee          5
    Makua             5
                 ...   
    William     3839236
    Michael     4312975
    Robert      4725713
    John        4845414
    James       4957166
    Name: Count, Length: 30274, dtype: int64



E ordenar...


```python
#In: 
(gb['Count'].
 sum().
 sort_values()
)
```




    Name
    Zyshonne          5
    Makenlee          5
    Makenlie          5
    Makinlee          5
    Makua             5
                 ...   
    William     3839236
    Michael     4312975
    Robert      4725713
    John        4845414
    James       4957166
    Name: Count, Length: 30274, dtype: int64



É comum, embora mais chato de ler, fazer tudo em uma única chamada. Isto é uma prática que vem do mundo SQL. A chamada abaixo seria o mesmo de:

```sql
SELECT Name, SUM(Count)
FROM baby_table
GROUPBY Name
ORDERBY SUM(Count)
```


```python
#In: 
(df[['Name', 'Count']].
 groupby('Name').
 sum().
 sort_values(by='Count')
)
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
      <th>Count</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Zyshonne</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Makenlee</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Makenlie</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Makinlee</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Makua</th>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>William</th>
      <td>3839236</td>
    </tr>
    <tr>
      <th>Michael</th>
      <td>4312975</td>
    </tr>
    <tr>
      <th>Robert</th>
      <td>4725713</td>
    </tr>
    <tr>
      <th>John</th>
      <td>4845414</td>
    </tr>
    <tr>
      <th>James</th>
      <td>4957166</td>
    </tr>
  </tbody>
</table>
<p>30274 rows × 1 columns</p>
</div>




```python
#In: 
(df[['Name', 'Year', 'Count']].
 groupby(['Name', 'Year']).
 sum()
)
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
      <th></th>
      <th>Count</th>
    </tr>
    <tr>
      <th>Name</th>
      <th>Year</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Aaban</th>
      <th>2013</th>
      <td>6</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Aadan</th>
      <th>2008</th>
      <td>12</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>6</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Zyriah</th>
      <th>2011</th>
      <td>6</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>5</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>7</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>6</td>
    </tr>
    <tr>
      <th>Zyshonne</th>
      <th>1998</th>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>548154 rows × 1 columns</p>
</div>



## NBA Salaries e Indexação Booleana

Por fim, vamos explorar alguns dados da NBA para entender a indexação booleana. Vamos carregar os dados da mesma forma que carregamos os dados dos nomes de crianças.


```python
#In: 
df = pd.read_csv('https://media.githubusercontent.com/media/icd-ufmg/material/master/aulas/03-Tabelas-e-Tipos-de-Dados/nba_salaries.csv')
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



Por fim, vamos indexar nosso DataFrame por booleanos. A linha abaixo pega um vetor de booleanos onde o nome do time é `Houston Rockets`.


```python
#In: 
df['TEAM'] == 'Houston Rockets'
```




    0      False
    1      False
    2      False
    3      False
    4      False
           ...  
    412    False
    413    False
    414    False
    415    False
    416    False
    Name: TEAM, Length: 417, dtype: bool



Podemos usar tal vetor para filtrar nosso DataFrame. A linha abaixo é o mesmo de um:

```sql
SELECT *
FROM table
WHERE TEAM = 'Houston Rockets'
```


```python
#In: 
filtro = df['TEAM'] == 'Houston Rockets'
df[filtro]
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
      <th>131</th>
      <td>Dwight Howard</td>
      <td>C</td>
      <td>Houston Rockets</td>
      <td>22.359364</td>
    </tr>
    <tr>
      <th>132</th>
      <td>James Harden</td>
      <td>SG</td>
      <td>Houston Rockets</td>
      <td>15.756438</td>
    </tr>
    <tr>
      <th>133</th>
      <td>Ty Lawson</td>
      <td>PG</td>
      <td>Houston Rockets</td>
      <td>12.404495</td>
    </tr>
    <tr>
      <th>134</th>
      <td>Corey Brewer</td>
      <td>SG</td>
      <td>Houston Rockets</td>
      <td>8.229375</td>
    </tr>
    <tr>
      <th>135</th>
      <td>Trevor Ariza</td>
      <td>SF</td>
      <td>Houston Rockets</td>
      <td>8.193030</td>
    </tr>
    <tr>
      <th>136</th>
      <td>Patrick Beverley</td>
      <td>PG</td>
      <td>Houston Rockets</td>
      <td>6.486486</td>
    </tr>
    <tr>
      <th>137</th>
      <td>K.J. McDaniels</td>
      <td>SG</td>
      <td>Houston Rockets</td>
      <td>3.189794</td>
    </tr>
    <tr>
      <th>138</th>
      <td>Terrence Jones</td>
      <td>PF</td>
      <td>Houston Rockets</td>
      <td>2.489530</td>
    </tr>
    <tr>
      <th>139</th>
      <td>Donatas Motiejunas</td>
      <td>PF</td>
      <td>Houston Rockets</td>
      <td>2.288205</td>
    </tr>
    <tr>
      <th>140</th>
      <td>Sam Dekker</td>
      <td>SF</td>
      <td>Houston Rockets</td>
      <td>1.646400</td>
    </tr>
    <tr>
      <th>141</th>
      <td>Clint Capela</td>
      <td>PF</td>
      <td>Houston Rockets</td>
      <td>1.242720</td>
    </tr>
    <tr>
      <th>142</th>
      <td>Montrezl Harrell</td>
      <td>PF</td>
      <td>Houston Rockets</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Assim como pegar os salários maior do que um certo valor!


```python
#In: 
df[df['SALARY'] > 20]
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
      <th>29</th>
      <td>Joe Johnson</td>
      <td>SF</td>
      <td>Brooklyn Nets</td>
      <td>24.894863</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Derrick Rose</td>
      <td>PG</td>
      <td>Chicago Bulls</td>
      <td>20.093064</td>
    </tr>
    <tr>
      <th>72</th>
      <td>LeBron James</td>
      <td>SF</td>
      <td>Cleveland Cavaliers</td>
      <td>22.970500</td>
    </tr>
    <tr>
      <th>131</th>
      <td>Dwight Howard</td>
      <td>C</td>
      <td>Houston Rockets</td>
      <td>22.359364</td>
    </tr>
    <tr>
      <th>156</th>
      <td>Chris Paul</td>
      <td>PG</td>
      <td>Los Angeles Clippers</td>
      <td>21.468695</td>
    </tr>
    <tr>
      <th>169</th>
      <td>Kobe Bryant</td>
      <td>SF</td>
      <td>Los Angeles Lakers</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>201</th>
      <td>Chris Bosh</td>
      <td>PF</td>
      <td>Miami Heat</td>
      <td>22.192730</td>
    </tr>
    <tr>
      <th>255</th>
      <td>Carmelo Anthony</td>
      <td>SF</td>
      <td>New York Knicks</td>
      <td>22.875000</td>
    </tr>
    <tr>
      <th>268</th>
      <td>Kevin Durant</td>
      <td>SF</td>
      <td>Oklahoma City Thunder</td>
      <td>20.158622</td>
    </tr>
  </tbody>
</table>
</div>



## Exercícios

Abaixo temos algumas chamadas em pandas. Tente explicar cada uma delas.


```python
#In: 
(df[['POSITION', 'SALARY']].
 groupby('POSITION').
 mean()
)
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
      <th>SALARY</th>
    </tr>
    <tr>
      <th>POSITION</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C</th>
      <td>6.082913</td>
    </tr>
    <tr>
      <th>PF</th>
      <td>4.951344</td>
    </tr>
    <tr>
      <th>PG</th>
      <td>5.165487</td>
    </tr>
    <tr>
      <th>SF</th>
      <td>5.532675</td>
    </tr>
    <tr>
      <th>SG</th>
      <td>3.988195</td>
    </tr>
  </tbody>
</table>
</div>




```python
#In: 
(df[['TEAM', 'SALARY']].
 groupby('TEAM').
 mean().
 sort_values('SALARY')
)
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
      <th>SALARY</th>
    </tr>
    <tr>
      <th>TEAM</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Phoenix Suns</th>
      <td>2.971813</td>
    </tr>
    <tr>
      <th>Utah Jazz</th>
      <td>3.095993</td>
    </tr>
    <tr>
      <th>Portland Trail Blazers</th>
      <td>3.246206</td>
    </tr>
    <tr>
      <th>Philadelphia 76ers</th>
      <td>3.267796</td>
    </tr>
    <tr>
      <th>Boston Celtics</th>
      <td>3.352367</td>
    </tr>
    <tr>
      <th>Milwaukee Bucks</th>
      <td>4.019873</td>
    </tr>
    <tr>
      <th>Detroit Pistons</th>
      <td>4.221176</td>
    </tr>
    <tr>
      <th>Toronto Raptors</th>
      <td>4.392507</td>
    </tr>
    <tr>
      <th>Brooklyn Nets</th>
      <td>4.408229</td>
    </tr>
    <tr>
      <th>Denver Nuggets</th>
      <td>4.459243</td>
    </tr>
    <tr>
      <th>Memphis Grizzlies</th>
      <td>4.466497</td>
    </tr>
    <tr>
      <th>Charlotte Hornets</th>
      <td>4.672355</td>
    </tr>
    <tr>
      <th>Indiana Pacers</th>
      <td>4.822694</td>
    </tr>
    <tr>
      <th>Atlanta Hawks</th>
      <td>4.969507</td>
    </tr>
    <tr>
      <th>New Orleans Pelicans</th>
      <td>5.032163</td>
    </tr>
    <tr>
      <th>Minnesota Timberwolves</th>
      <td>5.065186</td>
    </tr>
    <tr>
      <th>Los Angeles Clippers</th>
      <td>5.082624</td>
    </tr>
    <tr>
      <th>Washington Wizards</th>
      <td>5.296912</td>
    </tr>
    <tr>
      <th>New York Knicks</th>
      <td>5.338846</td>
    </tr>
    <tr>
      <th>Orlando Magic</th>
      <td>5.544567</td>
    </tr>
    <tr>
      <th>Dallas Mavericks</th>
      <td>5.978414</td>
    </tr>
    <tr>
      <th>Oklahoma City Thunder</th>
      <td>6.052010</td>
    </tr>
    <tr>
      <th>Sacramento Kings</th>
      <td>6.216808</td>
    </tr>
    <tr>
      <th>Los Angeles Lakers</th>
      <td>6.237086</td>
    </tr>
    <tr>
      <th>San Antonio Spurs</th>
      <td>6.511698</td>
    </tr>
    <tr>
      <th>Chicago Bulls</th>
      <td>6.568407</td>
    </tr>
    <tr>
      <th>Golden State Warriors</th>
      <td>6.720367</td>
    </tr>
    <tr>
      <th>Miami Heat</th>
      <td>6.794056</td>
    </tr>
    <tr>
      <th>Houston Rockets</th>
      <td>7.107153</td>
    </tr>
    <tr>
      <th>Cleveland Cavaliers</th>
      <td>10.231241</td>
    </tr>
  </tbody>
</table>
</div>


