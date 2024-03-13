---
layout: page
title: Causalidade (Incompleto)
nav_order: 1
---

# Causalidade
{: .no_toc .mb-2 }

Uma breve introdução ao santo graal da ciência de dados
{: .fs-6 .fw-300 }

{: .no_toc .text-delta }
Resultados Esperados

* Entender qual o problema de causalidade
* Entender a diferença entre estudos randomizados e observacionais

---
**Sumário**
1. TOC
{:toc}
---

Esta página ainda está sendo escrita. Por enquanto, leia as
referências em livro. Lembre-se que este material é uma
apostila (incompleta), não um livro!

Boa parte deste material introdutório é uma tradução do:
[Data8](https://inferentialthinking.com/chapters/01/what-is-data-science.html)

## Introdução

Ciência de Dados trata de tirar conclusões úteis de
conjuntos de dados grandes e diversos por meio de
exploração, previsão e inferência. A exploração envolve a
identificação de padrões nas informações. A previsão
envolve o uso de informações que conhecemos para fazer
suposições informadas sobre valores que gostaríamos de
saber. A inferência envolve quantificar o nosso grau de
certeza: os padrões que encontramos nos nossos dados também
aparecerão em novas observações? Quão precisas são nossas
previsões? Nossas principais ferramentas para exploração
são visualizações e estatísticas descritivas, para previsão
são aprendizado de máquina e otimização e para inferência
são testes e modelos estatísticos.

A estatística é um componente central da ciência de dados
porque a estatística estuda como tirar conclusões robustas
com base em informações incompletas. A computação é um
componente central porque a programação nos permite aplicar
técnicas de análise aos grandes e diversos conjuntos de
dados que surgem em aplicações do mundo real: não apenas
números, mas textos, imagens, vídeos e leituras de
sensores. Ciência de dados é tudo isso, mas é mais do que a
soma de suas partes por causa das aplicações. Ao
compreender um domínio específico, os cientistas de dados
aprendem a fazer perguntas apropriadas sobre seus dados e a
interpretar corretamente as respostas fornecidas por nossas
ferramentas inferenciais e computacionais.

## Experimentos Randomizados Controlados

O padrão outro da ciência de dados são os Experimentos
Randomizados Controlados (do inglês *Randomized Controlled
Experiment - RCT*). Em tais estudos, conseguimos separar
nossa base de dados em dois grupos: (1) o de controle; (2)
o de tratamento. Para tal, fazemos uso de um gerado de
números aleatórios, ou uma moeda, para separar os dados de
interesse nos grupos.

Para entender melhor, pense nos estudos da vacina de Covid.
Alguns indivíduos recebiam vacinas (o grupo de tratamento),
outros placebo (o grupo de controle).

Se você conseguir randomizar os indivíduos nos grupos de
tratamento e controle, estará realizando um experimento
controlado randomizado, também conhecido como ensaio
clínico randomizado (RCT). Às vezes, as respostas das
pessoas numa experiência são influenciadas pelo facto de
saberem em que grupo pertencem. Portanto, poderá querer
realizar uma experiência cega em que os indivíduos não
sabem se estão no grupo de tratamento ou no grupo de
controlo. Para que isso funcione, você terá que dar ao
grupo de controle um placebo, que é algo que se parece
exatamente com o tratamento, mas na verdade não tem efeito.

Experimentos controlados randomizados têm sido há muito
tempo um padrão ouro na área médica, por exemplo, para
estabelecer se um novo medicamento funciona. Eles também
estão se tornando mais comumente usados em outras áreas,
como a economia.

### Exemplo: Subsídios sociais no México.

Nas aldeias mexicanas da década de 1990, as crianças das
famílias pobres muitas vezes não estavam matriculadas na
escola. Um dos motivos era que os filhos mais velhos podiam
trabalhar e assim ajudar no sustento da família. Santiago
Levy, ministro do Ministério das Finanças mexicano, decidiu
investigar se os programas de assistência social poderiam
ser usados para aumentar as matrículas escolares e melhorar
as condições de saúde. Ele conduziu um ECR em um conjunto
de aldeias, selecionando algumas delas aleatoriamente para
receber um novo programa de bem-estar chamado PROGRESA. O
programa dava dinheiro às famílias pobres se os seus filhos
frequentassem a escola regularmente e a família utilizasse
cuidados de saúde preventivos. Foi dado mais dinheiro se as
crianças frequentassem a escola secundária do que a escola
primária, para compensar os salários perdidos das crianças,
e foi dado mais dinheiro às raparigas que frequentavam a
escola do que aos rapazes. As demais aldeias não receberam
esse tratamento e formaram o grupo controle. Devido à
randomização, não houve fatores de confusão e foi possível
constatar que o PROGRESA aumentou a escolarização. Para os
meninos, a matrícula aumentou de 73% no grupo de controle
para 77% no grupo PROGRESA. Para as meninas, o aumento foi
ainda maior, de 67% no grupo de controle para quase 75% no
grupo PROGRESA. Devido ao sucesso desta experiência, o
governo mexicano apoiou o programa com o novo nome
OPORTUNIDADES, como um investimento numa população saudável
e bem educada.

## Benefícios da Randomização

O método de randomização pode ser tão simples quanto jogar
uma moeda. Também pode ser um pouco mais complexo. Mas todo
método de randomização consiste em uma sequência de etapas
cuidadosamente definidas que permitem que as chances sejam
especificadas matematicamente. Isto tem duas consequências
importantes.

- Permite-nos explicar – matematicamente – a possibilidade
  de a aleatorização produzir grupos de tratamento e de
  controlo bastante diferentes entre si.

- Isso nos permite fazer afirmações matemáticas precisas
  sobre as diferenças entre os grupos de tratamento e de
  controle. Isto, por sua vez, ajuda-nos a tirar conclusões
  justificáveis sobre se o tratamento tem algum efeito.

## E se você não puder randomizar?

Em algumas situações pode não ser possível realizar uma
experiência aleatória controlada, mesmo quando o objetivo
é investigar a causalidade. Por exemplo, suponha que você
queira estudar os efeitos do consumo de álcool durante a
gravidez e designe aleatoriamente algumas mulheres grávidas
para o seu grupo de “álcool”. Você não deve esperar a
cooperação deles se lhes oferecer uma bebida. Em tais
situações, quase invariavelmente você estará conduzindo um
estudo observacional, não um experimento. Esteja alerta
para fatores de confusão.

Em casos como estes, realizamos estudos observacionais.

## Estudos Observacionais

Ler [aqui](https://inferentialthinking.com/chapters/02/causality-and-experiments.html).
