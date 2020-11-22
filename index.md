<<<<<<< HEAD
---
layout: home
title: Just the Class
nav_exclude: true
seo:
  type: Course
  name: Just the Class
---

# {{ site.tagline }}
{: .mb-2 }
{{ site.description }}
{: .fs-6 .fw-300 }

{% if site.announcements %}
{{ site.announcements.last }}
[Announcements](announcements.md){: .btn .btn-outline .fs-3 }
{% endif %}

## Just the Class

Just the Class is a GitHub Pages template developed for the purpose of quickly deploying course websites. In addition to serving plain web pages and files, it provides a boilerplate for:

- a [course calendar](calendar.md),
- a [staff](staff.md) page,
- and a weekly [schedule](schedule.md).

Just the Class is built on top of [Just the Docs](https://github.com/pmarsceill/just-the-docs), making it easy to extend for your own special use cases while providing sane defaults for most everything else. This means that you also get:

- automatic [navigation structure](https://pmarsceill.github.io/just-the-docs/docs/navigation-structure/),
- instant, full-text [search](https://pmarsceill.github.io/just-the-docs/docs/search/) and page indexing,
- and a small but powerful set of [UI components](https://pmarsceill.github.io/just-the-docs/docs/ui-components) and authoring [utilities](https://pmarsceill.github.io/just-the-docs/docs/utilities).

## Getting Started

Getting started with Just the Class is simple.

1. Create a [new repository based on Just the Class](https://github.com/kevinlin1/just-the-class/generate).
1. Update `_config.yml` and `index.md` with your course information.
1. Configure a [publishing source for GitHub Pages](https://help.github.com/en/articles/configuring-a-publishing-source-for-github-pages). Your course website is now live!
1. Edit and create `.md` [Markdown files](https://guides.github.com/features/mastering-markdown/) to add your content.

For a few open-source examples, see the following course websites and their source code.

- [CSE 390HA](https://courses.cs.washington.edu/courses/cse390ha/20au/) is an example of a single-page website: [source code](https://gitlab.cs.washington.edu/cse390ha/20au/website).
- [CSE 143](https://courses.cs.washington.edu/courses/cse143/20au/) hosts an entire online textbook with full-text search: [source code](https://gitlab.cs.washington.edu/cse143/20au/website).

Continue reading to learn how to setup a development environment on your local computer. This allows you to make incremental changes without directly modifying the live website.

### Local development environment

Just the Class is built for [Jekyll](https://jekyllrb.com), a static site generator. View the [quick start guide](https://jekyllrb.com/docs/) for more information. Just the Docs requires no special Jekyll plugins and can run on GitHub Pages' standard Jekyll compiler.

1. Follow the GitHub documentation for [Setting up your GitHub Pages site locally with Jekyll](https://help.github.com/en/articles/setting-up-your-github-pages-site-locally-with-jekyll).
1. Start your local Jekyll server.
```bash
$ bundle exec jekyll serve
```
1. Point your web browser to [http://localhost:4000](http://localhost:4000)
1. Reload your web browser after making a change to preview its effect.

For more information, refer to [Just the Docs](https://pmarsceill.github.io/just-the-docs/).
=======
|----------|------------|----------|----------------|
| [Slides] | [Material] | [Listas] | [Bibliografia] |

- - -

# DCC212: Introdução à Ciência dos Dados

Professores: Flavio Figueiredo

Departamento: Departamento de Ciência da Computação (DCC) - UFMG

## Material

### Parte Zero: Motivação (1 aula, reduzir não precisa de duas aqui)

#### Objetivos de Aprendizado

1. Motivar o curso e a carreira
1. Falar do grande problema de ciência de dados (causa e efeito)

#### Material

1. [Apresentação do Curso](https://github.com/icd-ufmg/material/blob/master/aulas/01-Apresentacao/Aula01-Apresentacao.ipynb)
1. [Causa e Efeito (Tratamento, Controle)](https://github.com/icd-ufmg/material/blob/master/aulas/02-Causa-e-Efeito/README.md)

### Parte Um: Análise Exploratória de Dados e Revisões (6 aulas, adicionar limpeza de dados)

Ao terminar esta parte do curso o discente deve saber o mínimo sobre como ler e plotar dados. Além do mais, deve ter feito
uma revisão do seu curso de Probabilidade (Probabilidade I) ou Probabilidade e Estatística.

#### Objetivos de Aprendizado

1. Aprender sobre tabelas de dados, csvs e tipos de colunas
1. Bons príncipios de visualização
1. Análise exploratória e limpeza e dados
1. Tendências Centrais
    1. Média, Mediana, Desvio Padrão etc.
1. Revisão de Probabilidade (pré-requisito do curso)
    1. Distribuições Discretas vs Contínuas
    1. A Normal
    1. Estimadores da média e sua variâncias

#### Material

1. [Tabelas e Tipos de Dados](https://github.com/icd-ufmg/material/blob/master/aulas/03-Tabelas-e-Tipos-de-Dados/Aula03-Tabelas.ipynb)
1. [Visualização de Dados](https://github.com/icd-ufmg/material/blob/master/aulas/04-EDA-e-Vis/Aula04-EDA-Vis.ipynb)
   * Sugiro também uma leitura do Capítulos 6.4, 6.5 e 6.6 do [Data100](https://www.textbook.ds100.org/). São novos e não foi possível adaptar para a aula. Material simples com princípios bem interessantes.
1. [Tendências Centrais](https://github.com/icd-ufmg/material/blob/master/aulas/05-Tendencias-Centrais/Aula05-Tendencias-Centrais.ipynb)
1. [Probabilidade](https://github.com/icd-ufmg/material/blob/master/aulas/06-Probabilidade/Aula06%20-%20Probabilidade.ipynb)
1. [Risco e Variância de Estimador](https://github.com/icd-ufmg/material/blob/master/aulas/07-Risco/Aula07%20-%20Risco.ipynb)

### Segunda Parte: Testes de Hipótese (6 aulas)

Ao terminar esta parte do curso o discente deve saber o mínimo sobre como o essencial de testes de hipóteses. Este curso
não cobre uma diversidade de testes. O foco maior é no entendimento de conceitos como: intervalos de confiança, valores p,
testes a/b e noções de assuntos avançados (poder e testes múltiplos). Além do mais, usamos o aracabouço de testes para
falar de ciência no geral vs ciência de dados.

#### Objetivos de Aprendizado

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

#### Material

1. [Teorema Central do Limite](TODO)
1. [Intervalos de Confiança](https://github.com/icd-ufmg/material/blob/master/aulas/09-ICs/09%20-%20Bootstrap.ipynb)
1. [Comparando Médias - Testes A/B](https://github.com/icd-ufmg/material/blob/master/aulas/10-AB/10%20-%20AB.ipynb)
1. [Testes de Hipóteses](https://github.com/icd-ufmg/material/blob/master/aulas/11-Hipoteses/11%20-%20Hipoteses.ipynb)
1. [Replicação e Método Científico](https://github.com/icd-ufmg/material/blob/master/aulas/13-CausalidadeRCT/13%20-%20Causalidade.ipynb)
1. [Fechamento Testes: Poder e Múltiplos](https://github.com/icd-ufmg/material/blob/master/aulas/12-Poder/12%20-%20Poder.ipynb)

### Terceira Parte: Correlação e Regressão (6 aulas, reorganizar)

Toda esta parte do curso foca apenas em Regressão. Embora pareça ser muitas aulas para o assunto, a ideia é seguir a filosofia do
curso de aprendizado de máquina do Andre Ng (Coursera). Regressão é usado não apenas como conceito estatístico, mas sim como uma forma
de apresentar o discente ao aprendizado de máquina.

#### Objetivos de Aprendizado

1. Correlação de Dados
1. Regressão Linear
1. Mínimos Quadrados
1. Verossimilhança

#### Material

1. [Correlação](https://github.com/icd-ufmg/material/blob/master/aulas/15-Correlacao/15%20-%20Correlacao.ipynb)
1. [Regressão Linear](https://github.com/icd-ufmg/material/blob/master/aulas/16-RegressaoLinear/16%20-%20Regressao%20Linear.ipynb)
1. [Verossimilhançca](https://github.com/icd-ufmg/material/blob/master/aulas/17-Verossimilhanca/17%20-%20Verossimilhanca.ipynb)
1. [Gradiente Descendente](https://github.com/icd-ufmg/material/blob/master/aulas/18-GradienteDescendente/18%20-%20Gradiente.ipynb)
1. [Regressão Múltipla](https://github.com/icd-ufmg/material/blob/master/aulas/19-Multipla/19%20-%20Multipla.ipynb)
1. [Treino, Validação e Testes (Sem Material, ver data100)](https://www.textbook.ds100.org/ch/15/bias_intro.html)

### Quarta Parte: Classificação e um Pouco de ML (6 aulas, reorganizar)

1. Regularização
1. Logística Parte 1
1. Logística Parte 2
1. KNN e Previsão na Prática
1. Aprendizado não Supervisionado (SVD e K-Means)
1. Ética e Ciência de Dados

## Bibliografia


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

[Slides]: https://drive.google.com/drive/folders/1ZIwHz7U8vKAgjvHwkL_R1hZlE_4dsmah?usp=sharing
[Informes]: #informes
[TPs]: #tps
[Bibliografia]: #bibliografia
[Material]: #material
[Exemplos]: ./aulas/
[Listas]: https://drive.google.com/open?id=11j-wgQ-MLn8Hj1fkYuFfkm3uinUxt1lq
>>>>>>> 705df652ebe5f2fb52ff49a01502714359be3481
