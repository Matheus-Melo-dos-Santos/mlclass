# Edição no dataset

# Valores a aleatórios e não normalizados

# Teste 1

## 1. Seleção

    - Remover todas as linhas com 3 ou mais itens faltantes

## 2. Processamento dos Dados

    - Excluir linhas que não contem insulina
    - Para os restantes, caso só falte uma coluna a ser preenchida, colocar um valor aleatório entre [min_coluna, max_coluna] 

<!-- Inserir imagem 01 -->
![resultado teste 1](img/ml%20-%20resultado%201.png)

# Teste 2

## 1. Seleção

    - Remover todas as linhas com 3 ou mais itens faltantes

## 2. Processamento dos Dados

    - Para os restantes, caso só falte uma coluna a ser preenchida, colocar um valor aleatório entre [min_coluna, max_coluna] 
    - Caso contrário (nenhuma das 2 for Insulina): preencher ambas
         com um valor aleatório entre [min_coluna, max_coluna] de cada
         coluna respectivamente.

<!-- Inserir imagem 02 -->
![resultado teste 2](img/ml%20-%20resultado%202.png)

# Teste 3

## 1. Seleção

    - Remover todas as linhas com 3 ou mais itens faltantes

## 2. Processamento dos Dados

    - Para os restantes colocar um valor aleatório entre [min_coluna, max_coluna]

![resultado teste 3](img/ml%20-%20resultado%203.png)

# Dados normalizados

# Teste 4 

    - Aplica os critérios do teste 2 (que teve o melhor resultado) com normalização MinMaxScaler

![resultado teste 4](img/ml%20-%20normalizado%20-%20MinMaxScaler.png)

# Teste 4 

    - Aplica os critérios do teste 2 (que teve o melhor resultado) com normalização StandardScale
    - StandardScale
        - Média: 0
        - Desvio padrão: 1

![resultado teste 6](img/ml%20-%20normalizado%20com%20StandardScaler.png)


