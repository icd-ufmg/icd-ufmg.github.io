---
layout: page
title: Tabelas e Tipos de Dados
nav_order: 2
---

[<img src="./colab_favicon_small.png" style="float: right;">](https://colab.research.google.com/github/icd-ufmg/icd-ufmg.github.io/blob/master/_lessons/02-tabelas.ipynb)


# Tabelas e Tipos de Dados

{: .no_toc .mb-2 }

Um breve resumo de alguns comandos python.
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

Neste notebook vamos explorar um pouco de dados tabulares. A principal biblioteca para leitura de dados tabulares em Python se chama **pandas**. A mesma é bastante poderosa implementando uma série de operações de bancos de dados (e.g., groupby e join). Nossa discussão será focada em algumas das funções principais do pandas que vamos explorar no curso. Existe uma série ampla de funcionalidades que a biblioteca (além de outras) vai trazer. 

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

plt.ion()
```

## Series

Existem dois tipos base de dados em pandas. O primeiro, Series, representa uma coluna de dados. Um combinação de Series vira um DataFrame (mais abaixo). Diferente de um vetor `numpy`, a Series de panda captura uma coluna de dados (ou vetor) indexado. Isto é, podemos nomear cada um dos valores.


```python
#In: 
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
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

    01-causalidade.md    08-amostragem.md	   15-linear.ipynb
    02-tabelas	     09-ics		   15-linear.md
    02-tabelas.ipynb     09-ics_files	   16-vero
    02-tabelas.md	     09-ics.ipynb	   16-vero_files
    03-viz		     09-ics.md		   16-vero.ipynb
    03-viz_files	     10-ab		   16-vero.md
    03-viz.ipynb	     10-ab_files	   17-gradiente
    03-viz.md	     10-ab.ipynb	   17-gradiente_files
    04-stat		     10-ab.md		   17-gradiente.ipynb
    04-stat_files	     11-hipoteses	   17-gradiente.md
    04-stat.ipynb	     11-hipoteses_files    18-multipla
    04-stat.md	     11-hipoteses.ipynb    18-multipla_files
    05-prob		     11-hipoteses.md	   18-multipla.ipynb
    05-prob_files	     12-causalidade	   18-multipla.md
    05-prob.ipynb	     12-causalidade_files  27-revisao
    05-prob.md	     12-causalidade.ipynb  27-revisao_files
    06-risco	     12-causalidade.md	   27-revisao.ipynb
    06-risco_files	     13-poder		   27-revisao.md
    06-risco.ipynb	     13-poder_files	   capital.json
    06-risco.md	     13-poder.ipynb	   colab_favicon.ico
    07-tcl		     13-poder.md	   colab_favicon.png
    07-tcl_files	     14-correlacao	   colab_favicon_small.png
    07-tcl.ipynb	     14-correlacao_files   compile_all.sh
    07-tcl.md	     14-correlacao.ipynb   compile_notebook.sh
    08-amostragem	     14-correlacao.md	   dom-casmurro.txt
    08-amostragem_files  15-linear
    08-amostragem.ipynb  15-linear_files


Com a opção -lha, mostramos meta-dados dos arquivos como o owner, tamanho e permissões. Note que todos os arquivos são .csv, isto é comma separated.


```python
#In: 
!ls -lha .
```

    total 12M
    drwxr-xr-x 38 flaviovdf flaviovdf 4.0K Feb  3 17:45 .
    drwxr-xr-x  9 flaviovdf flaviovdf 4.0K Feb  3 15:22 ..
    -rw-r--r--  1 flaviovdf flaviovdf  618 Feb  3 16:56 01-causalidade.md
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Feb  3 17:45 02-tabelas
    -rw-r--r--  1 flaviovdf flaviovdf 106K Feb  3 17:46 02-tabelas.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  46K Feb  3 17:36 02-tabelas.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Feb  3 15:56 03-viz
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Feb  3 15:56 03-viz_files
    -rw-r--r--  1 flaviovdf flaviovdf 759K Feb  3 17:39 03-viz.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  39K Feb  3 15:56 03-viz.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Feb  3 15:56 04-stat
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Feb  3 15:56 04-stat_files
    -rw-r--r--  1 flaviovdf flaviovdf 290K Feb  3 17:45 04-stat.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  26K Feb  3 15:56 04-stat.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Feb  3 15:57 05-prob
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Feb  3 15:57 05-prob_files
    -rw-r--r--  1 flaviovdf flaviovdf 885K Feb  3 17:42 05-prob.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  35K Feb  3 15:57 05-prob.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Feb  3 15:58 06-risco
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Feb  3 15:58 06-risco_files
    -rw-r--r--  1 flaviovdf flaviovdf 291K Feb  3 17:42 06-risco.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  15K Feb  3 15:58 06-risco.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Feb  3 15:58 07-tcl
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Feb  3 15:58 07-tcl_files
    -rw-r--r--  1 flaviovdf flaviovdf 447K Feb  3 17:42 07-tcl.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  14K Feb  3 15:58 07-tcl.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Feb  3 15:59 08-amostragem
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Feb  3 15:59 08-amostragem_files
    -rw-r--r--  1 flaviovdf flaviovdf 227K Feb  3 17:42 08-amostragem.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf 118K Feb  3 15:59 08-amostragem.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Feb  3 16:00 09-ics
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Feb  3 16:00 09-ics_files
    -rw-r--r--  1 flaviovdf flaviovdf 703K Feb  3 17:42 09-ics.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  88K Feb  3 16:00 09-ics.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Feb  3 16:04 10-ab
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Feb  3 16:04 10-ab_files
    -rw-r--r--  1 flaviovdf flaviovdf 172K Feb  3 17:42 10-ab.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  17K Feb  3 16:04 10-ab.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Feb  3 16:04 11-hipoteses
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Feb  3 16:04 11-hipoteses_files
    -rw-r--r--  1 flaviovdf flaviovdf 297K Feb  3 17:42 11-hipoteses.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf 105K Feb  3 16:04 11-hipoteses.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Feb  3 16:52 12-causalidade
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Feb  3 16:52 12-causalidade_files
    -rw-r--r--  1 flaviovdf flaviovdf  52K Feb  3 17:43 12-causalidade.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf 9.4K Feb  3 16:52 12-causalidade.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Feb  3 17:03 13-poder
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Feb  3 17:03 13-poder_files
    -rw-r--r--  1 flaviovdf flaviovdf 837K Feb  3 17:44 13-poder.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  25K Feb  3 17:03 13-poder.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Feb  3 16:49 14-correlacao
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Feb  3 16:49 14-correlacao_files
    -rw-r--r--  1 flaviovdf flaviovdf 1.8M Feb  3 17:43 14-correlacao.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  39K Feb  3 16:49 14-correlacao.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Feb  3 16:50 15-linear
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Feb  3 16:50 15-linear_files
    -rw-r--r--  1 flaviovdf flaviovdf 905K Feb  3 17:43 15-linear.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  25K Feb  3 16:50 15-linear.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Feb  3 16:52 16-vero
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Feb  3 16:52 16-vero_files
    -rw-r--r--  1 flaviovdf flaviovdf 1.0M Feb  3 17:43 16-vero.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  29K Feb  3 16:52 16-vero.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Feb  3 17:38 17-gradiente
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Feb  3 17:38 17-gradiente_files
    -rw-r--r--  1 flaviovdf flaviovdf 726K Feb  3 17:44 17-gradiente.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  61K Feb  3 17:38 17-gradiente.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Feb  3 17:00 18-multipla
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Feb  3 17:00 18-multipla_files
    -rw-r--r--  1 flaviovdf flaviovdf 670K Feb  3 17:44 18-multipla.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf 134K Feb  3 17:00 18-multipla.md
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Feb  2 08:09 27-revisao
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Feb  2 08:09 27-revisao_files
    -rw-r--r--  1 flaviovdf flaviovdf  63K Feb  3 17:44 27-revisao.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf  17K Feb  2 08:09 27-revisao.md
    -rw-r--r--  1 flaviovdf flaviovdf 172K Nov 30 09:22 capital.json
    -rw-r--r--  1 flaviovdf flaviovdf  72K Feb  3 17:08 colab_favicon.ico
    -rw-r--r--  1 flaviovdf flaviovdf 7.8K Feb  3 17:12 colab_favicon.png
    -rw-r--r--  1 flaviovdf flaviovdf 2.2K Feb  3 17:31 colab_favicon_small.png
    -rw-r--r--  1 flaviovdf flaviovdf   85 Dec 14 12:42 compile_all.sh
    -rw-r--r--  1 flaviovdf flaviovdf 1.1K Feb  3 17:37 compile_notebook.sh
    -rw-r--r--  1 flaviovdf flaviovdf 401K Nov 30 09:22 dom-casmurro.txt
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Feb  3 16:54 .ipynb_checkpoints
    -rw-r--r--  1 flaviovdf flaviovdf  29K Dec 19 15:46 .nbgrader.log


Vamos identificar qual a cara de um csv. O programa `head` imprime as primeiras `n` linhas de um arquivo.


```python
#In: 
!head baby.csv
```

    head: cannot open 'baby.csv' for reading: No such file or directory


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
gb.mean()
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


```python
#In: 
gb.sum()['Count'].sort_values()
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
gb.sum()['Count'].sort_values()
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
df.groupby('Name').sum().sort_values(by='Count')['Count']
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



Podemos inverter com ::-1


```python
#In: 
df.groupby(['Name', 'Year']).sum()
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
df[['POSITION', 'SALARY']].groupby('POSITION').mean()
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
df[['TEAM', 'SALARY']].groupby('TEAM').mean().sort_values('SALARY')
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



## Merge

Agora, vamos explorar algumas chamadas que fazem opereações de merge.


```python
#In: 
people = pd.DataFrame(
    [["Joey",      "blue",       42,  "M"],
     ["Weiwei",    "blue",       50,  "F"],
     ["Joey",      "green",       8,  "M"],
     ["Karina",    "green",  np.nan,  "F"],
     ["Fernando",  "pink",        9,  "M"],
     ["Nhi",       "blue",        3,  "F"],
     ["Sam",       "pink",   np.nan,  "M"]], 
    columns = ["Name", "Color", "Age", "Gender"])
people
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
      <th>Color</th>
      <th>Age</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Joey</td>
      <td>blue</td>
      <td>42.0</td>
      <td>M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Weiwei</td>
      <td>blue</td>
      <td>50.0</td>
      <td>F</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Joey</td>
      <td>green</td>
      <td>8.0</td>
      <td>M</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Karina</td>
      <td>green</td>
      <td>NaN</td>
      <td>F</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fernando</td>
      <td>pink</td>
      <td>9.0</td>
      <td>M</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Nhi</td>
      <td>blue</td>
      <td>3.0</td>
      <td>F</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sam</td>
      <td>pink</td>
      <td>NaN</td>
      <td>M</td>
    </tr>
  </tbody>
</table>
</div>




```python
#In: 
email = pd.DataFrame(
    [["Deb",  "deborah_nolan@berkeley.edu"],
     ["Sam",  np.nan],
     ["John", "doe@nope.com"],
     ["Joey", "jegonzal@cs.berkeley.edu"],
     ["Weiwei", "weiwzhang@berkeley.edu"],
     ["Weiwei", np.nan],
     ["Karina", "kgoot@berkeley.edu"]], 
    columns = ["User Name", "Email"])
email
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
      <th>User Name</th>
      <th>Email</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Deb</td>
      <td>deborah_nolan@berkeley.edu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sam</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>John</td>
      <td>doe@nope.com</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Joey</td>
      <td>jegonzal@cs.berkeley.edu</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Weiwei</td>
      <td>weiwzhang@berkeley.edu</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Weiwei</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Karina</td>
      <td>kgoot@berkeley.edu</td>
    </tr>
  </tbody>
</table>
</div>




```python
#In: 
people.merge(email, 
             how = "inner",
             left_on = "Name", right_on = "User Name")
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
      <th>Color</th>
      <th>Age</th>
      <th>Gender</th>
      <th>User Name</th>
      <th>Email</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Joey</td>
      <td>blue</td>
      <td>42.0</td>
      <td>M</td>
      <td>Joey</td>
      <td>jegonzal@cs.berkeley.edu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Joey</td>
      <td>green</td>
      <td>8.0</td>
      <td>M</td>
      <td>Joey</td>
      <td>jegonzal@cs.berkeley.edu</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Weiwei</td>
      <td>blue</td>
      <td>50.0</td>
      <td>F</td>
      <td>Weiwei</td>
      <td>weiwzhang@berkeley.edu</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Weiwei</td>
      <td>blue</td>
      <td>50.0</td>
      <td>F</td>
      <td>Weiwei</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Karina</td>
      <td>green</td>
      <td>NaN</td>
      <td>F</td>
      <td>Karina</td>
      <td>kgoot@berkeley.edu</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sam</td>
      <td>pink</td>
      <td>NaN</td>
      <td>M</td>
      <td>Sam</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


