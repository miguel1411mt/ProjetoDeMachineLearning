# Sistema inteligente de recomendação de culturas agrícolas

Projeto final de Machine Learning. Grupo: Miguel Torres e Mateo Zapelini.

Classificação multiclasse (22 culturas) com SVM, árvore de decisão, Random Forest e
Gradient Boosting, e agrupamento das condições de solo e clima com K-means.

Dataset: Crop Recommendation Dataset (Atharva Ingle, Kaggle),
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset,
salvo em `dados/Crop_recommendation.csv`.

## Como executar no Google Colab

1. Abra o notebook direto pelo link:
   https://colab.research.google.com/github/miguel1411mt/ProjetoDeMachineLearning/blob/main/projeto_final_recomendacao_culturas.ipynb
2. Clique em Ambiente de execução > Executar tudo.
3. Se o Colab avisar que o notebook não foi criado pelo Google, clique em Executar assim mesmo.

A primeira célula de código clona este repositório dentro do Colab, então não é preciso
enviar nenhum arquivo. Não precisa de GPU: o ambiente padrão (CPU) é suficiente.

Tempo estimado no Colab gratuito: cerca de 15 minutos. Quase todo esse tempo fica na
seção 3, na busca de hiperparâmetros do Gradient Boosting (cerca de 7 minutos) e na
validação cruzada repetida. A EDA e o K-means levam menos de 1 minuto cada. Enquanto uma
célula roda, o Colab mostra um ícone girando ao lado dela.

O Colab usa uma versão mais antiga do scikit-learn (1.6.1 em novembro de 2026). Com ela,
SVM, árvore de decisão, Random Forest e K-means dão exatamente os números do relatório.
O Gradient Boosting muda um pouco (F1 no teste 0,9939 em vez de 0,9940 e validação
cruzada repetida 0,9818 em vez de 0,9834), sem alterar as conclusões.

## Como executar no computador

Requer Python 3.12 ou mais recente.

    python -m pip install -r requirements.txt
    python 01_eda.py
    python 02_modelos_supervisionados.py
    python 03_kmeans_perfis.py

Ou abra `projeto_final_recomendacao_culturas.ipynb` no Jupyter e execute as células em
ordem. Tabelas e figuras são gravadas em `resultados/`. Em um computador com 4 a 8 núcleos,
o projeto completo leva de 3 a 8 minutos.

Resultados do relatório gerados com Python 3.13, numpy 2.5.3, pandas 3.0.5,
scikit-learn 1.9.1 e matplotlib 3.11.2.

O relatório está em `relatorio/`.

| Arquivo | Conteúdo |
| --- | --- |
| comum.py | Leitura dos dados, tradução e engenharia de atributos |
| 01_eda.py | Análise exploratória |
| 02_modelos_supervisionados.py | Pré-processamento, GridSearchCV, métricas, ROC e importância |
| 03_kmeans_perfis.py | K-means, escolha de K, perfis e estabilidade |
