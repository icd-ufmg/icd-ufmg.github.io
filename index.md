---
layout: home
title: DCC 212
nav_exclude: true
seo:
  type: Course
  name: Introdução à Ciência de Dados
---

# {{ site.tagline }}
{: .mb-2 }
{{ site.description }}
{: .fs-6 .fw-300 }

{% if site.announcements %}
{{ site.announcements.last }}
[Avisos](announcements.md){: .btn .btn-outline .fs-3 }
{% endif %}

---

## Sumário
{: .no_toc .text-delta }

1. TOC
{:toc}

---

{: .no_toc .mb-2 }

Uma introdução ao ciclo de ciência de dados em quatro partes.
{: .fs-6 .fw-300 }

O curso de Introdução à Ciência de Dados (DCC212) do DCC-UFMG tem como
principal objetivo trazer para os discentes um conhecimento estatístico através
de um ponto de vista computacional. O curso é fortemente inspirado nas ofertas
chamadas de Data8 e Data100 da universidade de Berkeley. Tais ementas (Data8 e
Data100) foram adaptadas para a realidade de discentes da graduação da UFMG. Em
particular, foi levado em conta que na nossa grade, os discentes já passaram
por matérias como: Álgebra Linear Computacional e Probabilidade.

Abaixo descrevemos as 4 partes (5 se contar a introdução) do curso junto com os
resultados de aprendizado esperados em cada. Tal estrutura em móudlos permite
que o aprendizado possa ser feito de diferentes fomas como:

Uma visão de um livro de estatística:
```
Mod 1 - Mod 2 - Mod 3 - Mod 4
```

Ou, uma visão mais focada em aprendizado de máquina.
```
Mod 1 - Mod 3 - Mod 4 - Mod 2
```

# Módulo 0: Motivação

Uma breve motivação em 1 aula.

**Objetivos de Aprendizado**

1. Motivar o curso e a carreira
1. Falar do grande problema de ciência de dados (causa e efeito)

# Módulo 1: Análise Exploratória

Ao terminar esta parte do curso o discente deve saber o mínimo sobre como ler e
plotar dados. Além do mais, deve ter feito uma revisão do seu curso de
Probabilidade (Probabilidade I) ou Probabilidade e Estatística.

**Objetivos de Aprendizado**

1. Aprender sobre tabelas de dados, csvs e tipos de colunas
1. Bons princípios de visualização
1. Análise exploratória e limpeza e dados
1. Tendências Centrais
    1. Média, Mediana, Desvio Padrão etc.
1. Revisão de Probabilidade (pré-requisito do curso)
    1. Distribuições Discretas vs Contínuas
    1. A Normal
    1. Estimadores da média e sua variâncias

# Módulo 2: Testes de Hipótese

Ao terminar esta parte do curso o discente deve saber o mínimo sobre como o
essencial de testes de hipóteses. Este curso não cobre uma diversidade de
testes. O foco maior é no entendimento de conceitos como: intervalos de
confiança, valores p, testes a/b e noções de assuntos avançados (poder e testes
múltiplos). Além do mais, usamos o arcabouço de testes para falar de ciência no
geral vs ciência de dados.

**Objetivos de Aprendizado**

1. Intervalos de Confiança
1. Bootstrap
1. Testes A/B
1. Valores P e Testes de Pemutação
    1. Seguindo a filosofia do Data8 e Data100 de Berkeley, não nos preocupamos muito em detalhes testes-t, wald etc.
       O foco é no conceito via métodos computacionais.
1. Valores P e Testes de Pemutação
    1. Seguindo a filosofia do Data8 e Data100 de Berkeley, não nos preocupamos muito em detalhes testes-t, wald etc.
       O foco é no conceito via métodos computacionais.
1. Ciência vs Ciência de Dados

# Módulo 3: Correlação e Regressão

Toda esta parte do curso foca apenas em Regressão. Embora pareça ser muitas
aulas para o assunto, a ideia é seguir a filosofia do curso de aprendizado de
máquina do Andrew Ng (Coursera). Regressão é usado não apenas como conceito
estatístico, mas sim como uma forma de apresentar o discente ao aprendizado de
máquina. Ou seja, aqui vamos explorar conceitos como funções de perda e
verossimilhança.

**Objetivos de Aprendizado**

1. Correlação de Dados
    1. Pearson e Spearman
1. Regressão Linear
    1. Qual o problema sendo resolvido
    1. Como fazer regressão múltipla e polinomial
1. Mínimos Quadrados
    1. Funções de Perda e Gradiente Descendente
1. Verossimilhança
   1. Funções de Ganho e Gradiente Ascendente
1. Regressão na Prática
    1. Engenharia de Atributos
    1. Introdução ao problema de previsão

# Módulo 4: Classificação e um Pouco de ML

Neste módulo o discente vai abordar o problema de inferência para problemas
de classificação. O módulo é uma continuação do anterior, sendo necessário
o aprendizado de Regressão antes de assistir as aulas daqui. O objetivo é
que com os módulos três e quatro, o discente aprenda diferentes formas de
relacionar variáveis explanatórias com respostas. Observe que tais relações
são complementares ao aprendizado do módulo um.

**Objetivos de Aprendizado**

1. Regularização e o Ciclo do Aprendizado de Máquina
    1. Falar de hiper-parâmetros e de treino, teste e validação.
1. Logística Parte 1
    1. Uma introdução a regressão logística via a função de verossimilhança
1. Logística Parte 2
    1. Logística na prática e entropia cruzada
1. KNN e Aprendizado na Prática
    1. Aula mais focada em fechar todo o ciclo, comparação entre KNNRegressor e
     Regressão Linear além de KNNClassifier e a Logística.
1. Aprendizado não Supervisionado (SVD e K-Means)
    1. SVD e PCA
    1. Kmeans

# Projeto

Com o [Projeto](/projeto) os discentes tem a chance de praticar o conhecimento
adquirido nos quatro módulos acima. É sugerida uma leitura detalhada do
material sobre o projeto neste sítio para um entendimento melhor de como
fazer uso de todo conhecimento de forma coerente.

# Bibliografia

  1. [Principles and Techniques of Data Science](https://www.textbook.ds100.org/) <br>
      Sam Lau, Joey Gonzalez, and Deb Nolan. <br>
     **Apenas em inglês. Aberto!**

  1. [Computational and Inferential Thinking: The Foundations of Data Science](http://www.inferentialthinking.com/) <br>
     Ani Adhikari and John DeNero <br>
     **Apenas em inglês. Aberto!**

  1. [Data Science from Scratch](http://shop.oreilly.com/product/0636920033400.do) <br>
     Joel Grus  <br>
     **Existe em Português!** Pago.

  1. [Fundamentos Estatísticos para Ciência da Computação](http://homepages.dcc.ufmg.br/~assuncao/EstatCC/FECD.pdf) <br>
     Renato Assunção <br>
     **Português**

  1. [An Introduction to Statistical Learning](www-bcf.usc.edu/~gareth/ISL/) <br>
      Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani <br>
     **Apenas em inglês. Aberto!**
