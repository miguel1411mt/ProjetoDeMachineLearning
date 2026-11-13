"""Etapa 5 (não supervisionada): perfis de ambiente de cultivo com K-means.

Pergunta: sem olhar a cultura, as 2.200 amostras de solo e clima formam
grupos de condições parecidas? Quais culturas caem em cada grupo?
O resultado ajuda a encontrar culturas que podem substituir umas às outras
em uma mesma condição de ambiente.
"""

import numpy as np  # Operações numéricas.
import pandas as pd  # Tabelas.
import matplotlib  # Gráficos.
matplotlib.use("Agg")  # Grava PNG sem abrir janelas.
import matplotlib.pyplot as plt  # Interface de desenho.
from sklearn.pipeline import Pipeline  # Sequência de preparação.
from sklearn.impute import SimpleImputer  # Ausentes (por segurança).
from sklearn.preprocessing import StandardScaler  # Mesma escala para todas as colunas.
from sklearn.cluster import KMeans  # Agrupamento.
from sklearn.decomposition import PCA  # Projeção em 2 dimensões para o gráfico.
from sklearn.metrics import silhouette_score, adjusted_rand_score
from comum import carregar_dados, ATRIBUTOS, SAIDA, SEMENTE

dados = carregar_dados()
X = dados[ATRIBUTOS]  # A cultura NÃO entra no agrupamento.
preparacao = Pipeline([("imputar", SimpleImputer(strategy="median")),
                       ("escala", StandardScaler())])  # K-means usa distância euclidiana.
Z = preparacao.fit_transform(X)  # Matriz 2200 x 7 padronizada.

# 1. Comparação de K = 2 a 15
avaliacoes, modelos = [], {}
for k in range(2, 16):
    km = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=SEMENTE)
    rot = km.fit_predict(Z)  # Grupo de cada amostra.
    avaliacoes.append([k, km.inertia_, silhouette_score(Z, rot)])
    modelos[k] = km
avaliacao = pd.DataFrame(avaliacoes, columns=["k", "inercia", "silhueta"])
print(avaliacao.round(3).to_string(index=False))
avaliacao.round(3).to_csv(SAIDA / "kmeans_escolha_k.csv", index=False)

fig, eixos = plt.subplots(1, 2, figsize=(11, 4))
eixos[0].plot(avaliacao.k, avaliacao.inercia, "o-")
eixos[0].set_xlabel("Número de grupos K"); eixos[0].set_ylabel("Inércia")
eixos[1].plot(avaliacao.k, avaliacao.silhueta, "o-")
eixos[1].axvline(7, color="gray", linestyle="--")  # K escolhido.
eixos[1].set_xlabel("Número de grupos K"); eixos[1].set_ylabel("Silhueta média")
fig.tight_layout()
fig.savefig(SAIDA / "kmeans_escolha_k.png", dpi=160)
plt.close(fig)

# 2. K escolhido. K=2 tem a maior silhueta, mas só separa maçã e uva do resto.
# Entre os valores úteis, a silhueta salta de 0,304 (K=6) para 0,333 (K=7)
# e depois fica estável; por isso adotamos K=7.
k_escolhido = 7
modelo = modelos[k_escolhido]
base = dados.copy()
base["grupo"] = modelo.labels_ + 1  # Numeração de 1 a 7.

# 3. Composição: quantas amostras de cada cultura caíram em cada grupo
composicao = pd.crosstab(base["cultura"], base["grupo"])
print("\nAmostras de cada cultura por grupo:\n", composicao)
composicao.to_csv(SAIDA / "kmeans_composicao.csv")
for g in composicao.columns:
    col = composicao[g]
    principais = col[col >= 20].sort_values(ascending=False).astype(int)  # Culturas com 20+ amostras.
    print(f"Grupo {g} ({col.sum()} amostras):", {c: int(v) for c, v in principais.items()})

# Mapa de calor da composição (% de cada cultura em cada grupo).
pct = composicao.div(composicao.sum(axis=1), axis=0) * 100
ordem = pct.idxmax(axis=1).sort_values().index  # Ordena culturas pelo grupo dominante.
pct = pct.loc[ordem]
fig, ax = plt.subplots(figsize=(7, 8))
im = ax.imshow(pct, cmap="Greens", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(k_escolhido), [f"G{g}" for g in pct.columns])
ax.set_yticks(range(len(pct)), pct.index)
for i in range(pct.shape[0]):
    for j in range(pct.shape[1]):
        if pct.iloc[i, j] >= 5:
            ax.text(j, i, f"{pct.iloc[i, j]:.0f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, label="% das amostras da cultura")
ax.set_title("Em qual grupo caem as amostras de cada cultura")
fig.tight_layout()
fig.savefig(SAIDA / "kmeans_composicao.png", dpi=160)
plt.close(fig)

# 4. Perfil de cada grupo nas unidades originais (medianas)
perfil = base.groupby("grupo")[ATRIBUTOS].median().round(1)
perfil.insert(0, "amostras", base["grupo"].value_counts().sort_index())
print("\nMedianas por grupo:\n", perfil.to_string())
perfil.to_csv(SAIDA / "kmeans_perfis.csv")

# 5. Concordância com as culturas e estabilidade
print("\nARI entre grupos e culturas:", round(adjusted_rand_score(base.cultura, base.grupo), 3))
ari_sementes = [adjusted_rand_score(modelo.labels_,
                KMeans(k_escolhido, n_init=1, random_state=s).fit_predict(Z)) for s in range(10)]
print("ARI com 10 sementes (n_init=1): mínimo", round(min(ari_sementes), 3),
      "| médio", round(float(np.mean(ari_sementes)), 3))
rng = np.random.default_rng(SEMENTE)
ari_sub = []
for _ in range(20):  # Subamostras com 80% das linhas.
    idx = rng.choice(len(Z), int(0.8 * len(Z)), replace=False)
    sub = KMeans(k_escolhido, n_init=20, random_state=SEMENTE).fit(Z[idx])
    ari_sub.append(adjusted_rand_score(modelo.labels_[idx], sub.labels_))
print("ARI em 20 subamostras de 80%: médio", round(float(np.mean(ari_sub)), 3),
      "| mínimo", round(float(np.min(ari_sub)), 3))

# 6. Visualização em 2 dimensões (PCA só para desenhar)
pca = PCA(n_components=2, random_state=SEMENTE)
P = pca.fit_transform(Z)
print("Variância explicada pelos 2 componentes:", pca.explained_variance_ratio_.sum().round(3))
fig, ax = plt.subplots(figsize=(7, 5.5))
cores = plt.cm.tab10(np.arange(k_escolhido))
for g in range(1, k_escolhido + 1):
    m = base.grupo.to_numpy() == g
    ax.scatter(P[m, 0], P[m, 1], s=8, color=cores[g - 1], label=f"Grupo {g}")
ax.set_xlabel("Componente principal 1"); ax.set_ylabel("Componente principal 2")
ax.set_title("Grupos do K-means projetados em 2 dimensões (PCA)")
ax.legend(fontsize=8, markerscale=2)
fig.tight_layout()
fig.savefig(SAIDA / "kmeans_pca.png", dpi=160)
plt.close(fig)

base.to_csv(SAIDA / "kmeans_amostras_com_grupo.csv", index=False)
