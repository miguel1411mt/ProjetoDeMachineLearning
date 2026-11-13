# Sistema inteligente de recomendação de culturas agrícolas

Projeto final de Machine Learning. Grupo: Miguel Torres e Mateo Zapelini.

Classificação multiclasse (22 culturas) com SVM, árvore de decisão, Random Forest e
Gradient Boosting, e agrupamento das condições de solo e clima com K-means.

Dataset: Crop Recommendation Dataset (Atharva Ingle, Kaggle),
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset,
salvo em `dados/Crop_recommendation.csv`.

## Como executar

    python -m pip install -r requirements.txt
    python 01_eda.py
    python 02_modelos_supervisionados.py
    python 03_kmeans_perfis.py

Ou abra `projeto_final_recomendacao_culturas.ipynb` (Jupyter ou Google Colab) e execute as
células em ordem. Tabelas e figuras são gravadas em `resultados/`.

Resultados gerados com Python 3.13, numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1 e
matplotlib 3.11.2. Não precisa de GPU.

| Arquivo | Conteúdo |
| --- | --- |
| comum.py | Leitura dos dados, tradução e engenharia de atributos |
| 01_eda.py | Análise exploratória |
| 02_modelos_supervisionados.py | Pré-processamento, GridSearchCV, métricas, ROC e importância |
| 03_kmeans_perfis.py | K-means, escolha de K, perfis e estabilidade |
