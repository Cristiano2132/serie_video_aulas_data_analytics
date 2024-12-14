# Plano de Videoaulas: Data Science e Analytics

## Módulo Introdutório: Fundamentos de Estatística

### 1. Estatísticas Descritivas

**Objetivo:** Introduzir medidas de tendência central, dispersão e gráficos.  
**Problema de Negócio:** Resumir um conjunto de dados de vendas para entender melhor as características dos clientes.  
**Background:**  
Estatísticas descritivas são essenciais para entender as características básicas de um conjunto de dados antes de avançar para análises complexas. Este passo é fundamental em qualquer pipeline de análise de dados. Medidas como média, mediana, variância e desvio padrão ajudam a resumir e visualizar os dados de forma compreensível.  
**Dados:** Dataset Iris do Scikit-Learn.  
**Exercício:** Calcular média, mediana, variância, desvio padrão e criar um histograma.

---

### 2. Relação entre Variáveis

**Objetivo:** Explorar relações entre variáveis qualitativas e quantitativas.  
**Problema de Negócio:** Identificar relações entre características dos clientes e suas decisões de compra.  
**Background:**  
A relação entre variáveis ajuda a identificar associações importantes em dados. O coeficiente de correlação mede a força e a direção de uma relação linear entre variáveis numéricas, enquanto o teste qui-quadrado avalia associações entre categorias.  
**Exercício:**

- Usar `scipy.stats` para calcular o coeficiente de correlação de Pearson.

```python
from scipy.stats import pearsonr

# Exemplo de dados
x = [10, 20, 30, 40]
y = [15, 25, 35, 45]

# Coeficiente de correlação
corr, _ = pearsonr(x, y)
print(f"Coeficiente de correlação: {corr:.2f}")
```

**Fontes de Dados:** Kaggle, UCI Machine Learning Repository

### 3. Distribuições de Probabilidade

**Objetivo:** Apresentar distribuições discretas e contínuas.  
**Problema de Negócio:** Simular vendas diárias de uma loja usando distribuições binomial e normal.  
**Background:**  
Compreender distribuições é essencial para modelar incertezas em dados. A distribuição normal é amplamente usada por sua aplicação em fenômenos naturais, enquanto a binomial é útil para modelar contagens de sucessos em ensaios.  
**Exercício:**

- Usar numpy para simular dados de uma distribuição normal.

```python
import numpy as np

# Gerar dados normalmente distribuídos
data = np.random.normal(loc=50, scale=10, size=1000)
```

**Fontes de Dados:** Simulação de dados com NumPy

### 4. Testes de Hipóteses

**Objetivo:** Introduzir testes estatísticos para médias, variâncias e associações.  
**Problemas de Negócio:**

1. Verificar se a média de vendas de uma loja é diferente de um valor esperado.
2. Comparar a média de duas lojas para avaliar diferenças de performance.  
   **Background:**  
   Os testes de hipóteses ajudam a validar suposições sobre os dados. O teste Z é usado para grandes amostras, enquanto o teste t é apropriado para amostras menores. O teste qui-quadrado avalia associações em tabelas de frequência.  
   **Exercícios Comuns e Exemplos:**

3. Teste Z para médias (uma amostra): Verificar se a média de vendas é igual a 500 unidades.

```python
from statsmodels.stats.weightstats import ztest

# Dados fictícios
sales = [510, 490, 505, 515, 480]

# Teste Z
z_stat, p_val = ztest(sales, value=500)
print(f"Estatística Z: {z_stat:.2f}, p-valor: {p_val:.4f}")
```

2. Teste t para médias (uma amostra): Comparar a média de vendas com 500 unidades.

```python
from scipy.stats import ttest_1samp

# Teste t
t_stat, p_val = ttest_1samp(sales, 500)
print(f"Estatística t: {t_stat:.2f}, p-valor: {p_val:.4f}")
```

3. Teste t para duas amostras independentes: Comparar médias de vendas entre duas lojas.

```python
from scipy.stats import ttest_ind

# Dados fictícios
store1 = [510, 520, 530, 500]
store2 = [480, 490, 495, 500]

# Teste t
t_stat, p_val = ttest_ind(store1, store2)
print(f"Estatística t: {t_stat:.2f}, p-valor: {p_val:.4f}")
```

4. Teste F para variâncias: Comparar variâncias de vendas entre duas lojas.

```python
from scipy.stats import f_oneway

# Teste F
f_stat, p_val = f_oneway(store1, store2)
print(f"Estatística F: {f_stat:.2f}, p-valor: {p_val:.4f}")
```

5. Teste qui-quadrado para tabelas de frequência: Verificar a associação entre categorias.

```python
from scipy.stats import chi2_contingency

# Tabela de contingência
table = [[10, 20], [30, 40]]

# Teste qui-quadrado
chi2, p_val, dof, expected = chi2_contingency(table)
print(f"Qui-quadrado: {chi2:.2f}, p-valor: {p_val:.4f}")
```

6. Intervalo de Confiança: Construir um intervalo de confiança para a média.

```python
import scipy.stats as stats

mean = np.mean(sales)
sem = stats.sem(sales)
ci = stats.t.interval(alpha=0.95, df=len(sales)-1, loc=mean, scale=sem)
print(f"Intervalo de Confiança: {ci}")
```

**Fontes de Dados:** Dados fictícios ou datasets de vendas de lojas (Kaggle, UCI Machine Learning Repository)

## Módulo: Modelos Supervisionados e Não Supervisionados de Machine Learning

### 1. Unsupervised Learning: Clustering

**Objetivo:** Agrupar clientes com base em comportamento.  
**Problema de Negócio:** Aplicar K-means para segmentar clientes de um e-commerce.  
**Background:**  
Segmentar clientes ajuda empresas a personalizar ofertas e campanhas. K-means é uma técnica eficiente para identificar padrões em grandes conjuntos de dados.  
**Dados:** Mall Customers Dataset, disponível no Kaggle.

### 2. Unsupervised Learning: PCA

**Objetivo:** Reduzir a dimensionalidade de dados complexos.  
**Problema de Negócio:** Aplicar PCA em um dataset de atributos físicos de flores.  
**Background:**  
O PCA é usado para simplificar conjuntos de dados ao identificar direções de máxima variância, reduzindo ruído e melhorando a eficiência computacional.  
**Dados:** Dataset Iris do Scikit-Learn.

### 3. Supervised Learning: Modelos Logísticos

**Objetivo:** Prever inadimplência em operações financeiras.  
**Problema de Negócio:** Construir um modelo de regressão logística binária para prever a probabilidade de um cliente ser inadimplente.  
**Background:**  
Modelos logísticos são amplamente usados para prever a probabilidade de um evento binário, como aprovação de crédito ou fraude.  
**Dados:** Credit Risk Dataset, disponível no Kaggle.

### 4. Supervised Learning: Modelos para Dados de Contagem

**Objetivo:** Modelar frequências de eventos.  
**Problema de Negócio:** Usar regressão de Poisson para prever ocorrências de incidentes de segurança no trabalho.  
**Background:**  
A regressão de Poisson é ideal para modelar contagens de eventos em um intervalo de tempo fixo.  
**Dados:** Simular um dataset de dados de incidentes.

### 5. Séries Temporais

**Objetivo:** Analisar padrões temporais e fazer previsões.  
**Problema de Negócio:** Criar um modelo ARIMA para prever vendas semanais de uma loja.  
**Background:**  
Séries temporais são usadas para analisar dados que variam no tempo, como vendas, tráfego ou clima. Modelos ARIMA são eficazes para previsão em séries estacionárias.  
**Dados:** Retail Sales Dataset, disponível no Kaggle.

## Módulo: Tendências em Data Science e Analytics

### 1. Data Wrangling

**Objetivo:** Manipular dados com eficiência.  
**Problema de Negócio:** Limpar e preparar dados de vendas para análise.  
**Background:**  
Data wrangling é o processo de organizar e formatar dados para análise. É um passo crucial para garantir resultados precisos.  
**Ferramentas:** Pandas e Jupyter Notebooks.  
**Dados:** Vendas de e-commerce (Kaggle, UCI Machine Learning Repository).

### 2. Text Mining e Sentiment Analysis

**Objetivo:** Identificar sentimentos em textos.  
**Problema de Negócio:** Analisar reviews da Amazon para identificar opiniões positivas e negativas.  
**Background:**  
A análise de sentimentos é amplamente usada para entender percepções dos clientes sobre produtos e serviços. Utilizando técnicas de mineração de texto, é possível extrair informações valiosas de grandes volumes de dados textuais.  
**Dados:** Amazon Reviews Dataset, disponível no Kaggle.

### 3. Deep Learning

**Objetivo:** Introduzir redes neurais artificiais.  
**Problema de Negócio:** Criar uma rede neural para classificação de imagens de dígitos escritos à mão.  
**Background:**  
Redes neurais são inspiradas no cérebro humano e são usadas para resolver problemas complexos como reconhecimento de imagem e processamento de linguagem natural. Deep learning tem se destacado em várias áreas devido à sua capacidade de aprender padrões complexos a partir de grandes quantidades de dados.  
**Dados:** MNIST Dataset, disponível no Kaggle.

## Exemplo de Estrutura das Aulas

### Aula 1: Introdução às Estatísticas Descritivas

1. **Objetivo:** Compreender as principais medidas de tendência central e dispersão.
2. **Conteúdo:** Conceitos de média, mediana, variância, desvio padrão e histogramas.
3. **Exercício Prático:** Usar Python (bibliotecas como NumPy e Pandas) para calcular estatísticas descritivas e criar histogramas com o dataset Iris.

### Aula 2: Exploração da Relação entre Variáveis

1. **Objetivo:** Identificar e medir relações entre variáveis numéricas e categóricas.
2. **Conteúdo:** Coeficiente de correlação de Pearson, testes qui-quadrado.
3. **Exercício Prático:** Calcular correlações e realizar testes qui-quadrado com dados fictícios.

### Aula 3: Distribuições de Probabilidade

1. **Objetivo:** Conhecer distribuições de probabilidade discretas e contínuas.
2. **Conteúdo:** Distribuições binomial e normal.
3. **Exercício Prático:** Simular dados utilizando distribuições e visualizar resultados com gráficos.

### Aula 4: Testes de Hipóteses

1. **Objetivo:** Aplicar testes estatísticos para validar suposições sobre dados.
2. **Conteúdo:** Teste Z, teste t, teste F, teste qui-quadrado, intervalos de confiança.
3. **Exercício Prático:** Realizar testes de hipóteses com dados fictícios e interpretar resultados.

### Aula 5: Aprendizado Não Supervisionado – Clustering

1. **Objetivo:** Aplicar técnicas de agrupamento para segmentação de dados.
2. **Conteúdo:** Algoritmo K-means.
3. **Exercício Prático:** Segmentar clientes de um e-commerce utilizando K-means com um dataset disponível no Kaggle.

### Aula 6: Aprendizado Não Supervisionado – PCA

1. **Objetivo:** Reduzir a dimensionalidade de dados complexos.
2. **Conteúdo:** Algoritmo PCA (Análise de Componentes Principais).
3. **Exercício Prático:** Aplicar PCA em dados do dataset Iris para reduzir a dimensionalidade e visualizar os resultados.

### Aula 7: Aprendizado Supervisionado – Regressão Logística

1. **Objetivo:** Construir um modelo de regressão logística para previsões binárias.
2. **Conteúdo:** Conceitos de regressão logística, interpretação de coeficientes.
3. **Exercício Prático:** Prever inadimplência financeira com um dataset de risco de crédito do Kaggle.

### Aula 8: Aprendizado Supervisionado – Regressão de Poisson

1. **Objetivo:** Modelar contagens de eventos usando regressão de Poisson.
2. **Conteúdo:** Conceitos de regressão de Poisson, adequação do modelo.
3. **Exercício Prático:** Prever incidentes de segurança no trabalho com dados simulados.

### Aula 9: Séries Temporais

1. **Objetivo:** Analisar dados temporais e fazer previsões.
2. **Conteúdo:** Modelos ARIMA.
3. **Exercício Prático:** Prever vendas semanais de uma loja usando um dataset de vendas no Kaggle.

### Aula 10: Data Wrangling

1. **Objetivo:** Limpar e preparar dados para análise.
2. **Conteúdo:** Técnicas de manipulação de dados com Pandas.
3. **Exercício Prático:** Limpar e preparar dados de vendas de e-commerce para análise.

### Aula 11: Text Mining e Sentiment Analysis

1. **Objetivo:** Realizar análise de sentimentos em grandes volumes de textos.
2. **Conteúdo:** Técnicas de mineração de texto e análise de sentimentos.
3. **Exercício Prático:** Analisar reviews da Amazon para identificar sentimentos usando um dataset disponível no Kaggle.

### Aula 12: Deep Learning

1. **Objetivo:** Introduzir conceitos e aplicações de redes neurais.
2. **Conteúdo:** Estrutura de redes neurais, treinamento e avaliação de modelos.
3. **Exercício Prático:** Classificar dígitos escritos à mão usando o dataset MNIST.
