# Predição de Valores de Aluguel

Este projeto tem como objetivo construir um modelo preditivo capaz de **estimar o valor de aluguel de imóveis** a partir de variáveis como localização, características do imóvel e atributos adicionais fornecidos no conjunto de dados.

## 1. Objetivo do Projeto

O objetivo central é oferecer modelos que auxiliem na precificação de imóveis, permitindo:

- Identificar fatores que mais influenciam o preço do aluguel;  
- Prever de forma confiável o valor de aluguel de novos imóveis;  
- Comparar diferentes algoritmos e estratégias de modelagem;  
- Apoiar tomadas de decisão no mercado imobiliário.  

## 2. Preparação dos Dados

- Criação da variável-alvo transformada `log_rent` (logaritmo do valor do aluguel) para corrigir assimetria;  
- Tratamento de valores ausentes;  
- Codificação de variáveis categóricas e padronização de variáveis numéricas;  
- Análise exploratória (EDA) para identificar outliers e padrões de distribuição;  
- Divisão dos dados em treino e teste para avaliação dos modelos.  

## 3. Resultados

Foram avaliados os seguintes modelos de regressão:

- Regressão Linear (OLS)  
- Ridge e Lasso Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  

As métricas consideradas foram **R²**, **MAE** e **RMSE**, tanto na escala logarítmica quanto na escala original dos valores de aluguel.  

| Modelo                 | R²     | MAE (R$) | RMSE (R$) | Observações |
|-------------------------|--------|----------|-----------|-------------|
| Regressão Linear (rent) | 0.625  | 883.50   | 1,397.46  | Sem log-transform |
| Regressão Linear (log)  | 0.710  | 1,306.20 | 0.33 (log)| Melhor ajuste que sem log |
| Decision Tree           | 0.9934 | 67.45    | 182.87    | Forte overfitting |
| Random Forest           | 0.9966 | 35.58    | 131.32    | Melhor equilíbrio |
| Gradient Boosting       | 0.9916 | 116.48   | 205.67    | Bom desempenho |

Destaque: **Random Forest** apresentou o melhor equilíbrio entre desempenho e generalização, alcançando **R² ≈ 0.997** e erro médio absoluto em torno de **R$ 35**.  

## 4. Estratégias de Modelagem

- Análise exploratória detalhada e tratamento da variável resposta;  
- Comparação entre modelos lineares e baseados em árvores;  
- Avaliação com métricas padronizadas;  
- Interpretação de resíduos e análise de variáveis mais importantes.  

## 5. Ferramentas Utilizadas

- Python 3.12;  
- Jupyter Notebook (execução via VSCode);  
- Bibliotecas:
  - `pandas` e `numpy` para manipulação de dados;  
  - `matplotlib` e `seaborn` para visualização;  
  - `scikit-learn` e `statsmodels` para modelagem;  

## 6. Possíveis Melhorias

- Implementação de um `Pipeline` completo com `ColumnTransformer`;  
- Aplicação de validação cruzada para maior robustez;  
- Inclusão de técnicas de seleção de variáveis e análise de multicolinearidade;  
- Testes com algoritmos adicionais (XGBoost, LightGBM, CatBoost);  
- Desenvolvimento de um modelo deployável (API ou dashboard interativo).  

## Autor

**Ricardo Luís Bertolucci Filho**  

- [LinkedIn](https://www.linkedin.com/in/ricardo-lu%C3%ADs-bertolucci-filho/)  
- [GitHub](https://github.com/ric-rky/ric-rky)  
- E-mail: bertolucci.rl@gmail.com  

##

Notebook principal: [`projeto_aluguel.ipynb`](https://github.com/ric-rky/Predicao-de-valores-de-aluguel/blob/main/projeto_aluguel.ipynb)
