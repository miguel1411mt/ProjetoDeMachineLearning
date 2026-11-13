"""Etapa 1: análise exploratória de dados (EDA).

Responde: quantas linhas e colunas existem, quais são os tipos, quantos
valores ausentes, quantos valores extremos e se as classes estão equilibradas.
"""

import numpy as np  # Operações numéricas.
import pandas as pd  # Tabelas.
import matplotlib  # Gráficos.
matplotlib.use("Agg")  # Grava PNG sem abrir janelas.
import matplotlib.pyplot as plt  # Interface de desenho.
from comum import carregar_dados, ATRIBUTOS, SAIDA

dados = carregar_dados()  # 2200 linhas x 8 colunas.

# a) Descrição geral
print("Linhas e colunas:", dados.shape)
print("\nTipos de dados:\n", dados.dtypes)
resumo = dados[ATRIBUTOS].describe().T  # Média, desvio, mínimo, quartis e máximo.
print("\nResumo estatístico:\n", resumo.round(2))
resumo.round(2).to_csv(SAIDA / "eda_resumo.csv")

# b) Diagnóstico: ausentes, duplicadas, outliers e balanceamento
print("\nValores ausentes por coluna:\n", dados.isna().sum())
print("Linhas duplicadas:", dados.duplicated().sum())

outliers = {}  # Contagem de valores fora de 1,5 x IQR em cada coluna.
for coluna in ATRIBUTOS:
    q1, q3 = dados[coluna].quantile([0.25, 0.75])  # Primeiro e terceiro quartis.
    iqr = q3 - q1  # Amplitude interquartil.
    fora = (dados[coluna] < q1 - 1.5 * iqr) | (dados[coluna] > q3 + 1.5 * iqr)
    outliers[coluna] = int(fora.sum())
tabela_out = pd.Series(outliers, name="outliers_iqr")
tabela_out = pd.concat([tabela_out, (tabela_out / len(dados) * 100).round(1).rename("pct")], axis=1)
print("\nOutliers pela regra de 1,5 x IQR:\n", tabela_out)
tabela_out.to_csv(SAIDA / "eda_outliers.csv")

# Quais culturas concentram os outliers de potássio e de chuva?
for coluna in ["potassio", "chuva_mm", "fosforo"]:
    q1, q3 = dados[coluna].quantile([0.25, 0.75])
    limite = q3 + 1.5 * (q3 - q1)  # Limite superior.
    print(f"Culturas com {coluna} acima de {limite:.1f}:",
          dados[dados[coluna] > limite]["cultura"].value_counts().to_dict())

contagem = dados["cultura"].value_counts()  # Amostras por cultura.
print("\nAmostras por cultura: mínimo", contagem.min(), "| máximo", contagem.max(),
      "| razão maior/menor:", contagem.max() / contagem.min())

# c) Visualizações
fig, eixos = plt.subplots(2, 4, figsize=(14, 6))  # Um histograma por atributo.
for eixo, coluna in zip(eixos.ravel(), ATRIBUTOS):
    eixo.hist(dados[coluna], bins=30, color="#5b8c5a")
    eixo.set_title(coluna)
eixos.ravel()[-1].axis("off")  # Sobra um quadro vazio.
fig.suptitle("Distribuição dos atributos (2.200 amostras)")
fig.tight_layout()
fig.savefig(SAIDA / "eda_histogramas.png", dpi=160)
plt.close(fig)

correlacao = dados[ATRIBUTOS].corr()  # Correlação de Pearson.
print("\nCorrelação entre atributos:\n", correlacao.round(2))
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(correlacao, cmap="RdBu_r", vmin=-1, vmax=1)  # Mapa de calor.
ax.set_xticks(range(7), ATRIBUTOS, rotation=45, ha="right")
ax.set_yticks(range(7), ATRIBUTOS)
for i in range(7):
    for j in range(7):
        ax.text(j, i, f"{correlacao.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im)
ax.set_title("Correlação entre os atributos")
fig.tight_layout()
fig.savefig(SAIDA / "eda_correlacao.png", dpi=160)
plt.close(fig)

# Boxplot por cultura para os dois atributos com mais outliers.
ordem = dados.groupby("cultura")["potassio"].median().sort_values().index
fig, eixos = plt.subplots(2, 1, figsize=(13, 8))
for eixo, coluna in zip(eixos, ["potassio", "chuva_mm"]):
    grupos = [dados.loc[dados.cultura == c, coluna] for c in ordem]
    eixo.boxplot(grupos, tick_labels=list(ordem))
    eixo.set_ylabel(coluna)
    eixo.tick_params(axis="x", rotation=60)
fig.suptitle("Potássio e chuva por cultura: os 'outliers' são culturas inteiras")
fig.tight_layout()
fig.savefig(SAIDA / "eda_boxplot_culturas.png", dpi=160)
plt.close(fig)

medianas = dados.groupby("cultura")[ATRIBUTOS].median().round(1)  # Perfil típico.
medianas.to_csv(SAIDA / "eda_medianas_por_cultura.csv")
print("\nMedianas por cultura:\n", medianas)
