"""Etapas 2 a 4: pré-processamento, treino, otimização e avaliação.

Problema: classificação multiclasse. A partir de N, P, K, temperatura,
umidade, pH e chuva, prever qual das 22 culturas é a mais indicada.
Modelos: SVM (modelo A), árvore de decisão (modelo B) e dois ensembles
(Random Forest e Gradient Boosting).
"""

import time  # Mede o tempo de treino.
import numpy as np  # Operações numéricas.
import pandas as pd  # Tabelas.
import matplotlib  # Gráficos.
matplotlib.use("Agg")  # Grava PNG sem abrir janelas.
import matplotlib.pyplot as plt  # Interface de desenho.
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline  # Encadeia as etapas.
from sklearn.preprocessing import StandardScaler, FunctionTransformer, label_binarize
from sklearn.impute import SimpleImputer  # Tratamento de ausentes.
from sklearn.svm import SVC  # Modelo A.
from sklearn.calibration import CalibratedClassifierCV  # Probabilidades para o SVM.
from sklearn.tree import DecisionTreeClassifier, export_text  # Modelo B.
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance  # Importância por permutação.
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
from comum import carregar_dados, criar_atributos, ATRIBUTOS, SAIDA, SEMENTE

# ------------------------------------------------------------------
# 1. Dados e divisão 70% treino, 15% validação, 15% teste
# ------------------------------------------------------------------
dados = carregar_dados()
X = dados[ATRIBUTOS]  # Sete atributos numéricos.
y = dados["cultura"]  # Alvo com 22 classes.

X_tr, X_resto, y_tr, y_resto = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEMENTE)  # 70% para treino.
X_val, X_te, y_val, y_te = train_test_split(
    X_resto, y_resto, test_size=0.50, stratify=y_resto, random_state=SEMENTE)  # 15% + 15%.
print("Treino:", X_tr.shape, "| validação:", X_val.shape, "| teste:", X_te.shape)
# stratify mantém 70, 15 e 15 amostras de cada cultura em cada parte.

# ------------------------------------------------------------------
# 2. Pré-processamento dentro do Pipeline
# ------------------------------------------------------------------
# - SimpleImputer(median): o dataset não tem ausentes, mas a etapa protege o
#   sistema contra análises de solo incompletas no uso real.
# - "atributos": passthrough (só os 7 originais) ou criar_atributos (+4 colunas).
#   O GridSearchCV decide se a engenharia de atributos ajuda.
# - StandardScaler: obrigatório no SVM, que usa distâncias. Nas árvores não muda
#   os cortes, mas foi mantido para que todos os modelos recebam a mesma entrada.
engenharia = FunctionTransformer(criar_atributos)  # Acrescenta proporções N:P:K.


def montar(modelo):
    """Cria o Pipeline padrão com o modelo informado na última etapa."""
    return Pipeline([
        ("atributos", "passthrough"),
        ("imputar", SimpleImputer(strategy="median")),
        ("escala", StandardScaler()),
        ("modelo", modelo),
    ])


opcoes_atributos = ["passthrough", engenharia]  # Testadas em todos os modelos.

# ------------------------------------------------------------------
# 3. Modelos e grades de hiperparâmetros
# ------------------------------------------------------------------
buscas = {
    "SVM": (montar(SVC(random_state=SEMENTE)), {
        "atributos": opcoes_atributos,
        "modelo__kernel": ["linear", "rbf"],
        "modelo__C": [0.1, 1, 10, 100],
        "modelo__gamma": ["scale", 0.01, 0.1, 1],
    }),
    "Árvore de decisão": (montar(DecisionTreeClassifier(random_state=SEMENTE)), {
        "atributos": opcoes_atributos,
        "modelo__criterion": ["gini", "entropy"],
        "modelo__max_depth": [4, 6, 8, 10, 12, None],
        "modelo__min_samples_leaf": [1, 2, 4, 8],
        "modelo__ccp_alpha": [0.0, 0.001, 0.005],
    }),
    "Random Forest": (montar(RandomForestClassifier(random_state=SEMENTE, n_jobs=-1)), {
        "atributos": opcoes_atributos,
        "modelo__n_estimators": [100, 300],
        "modelo__max_features": ["sqrt", 0.5],
        "modelo__min_samples_leaf": [1, 3],
    }),
    "Gradient Boosting": (montar(GradientBoostingClassifier(
            n_estimators=150, subsample=0.8, random_state=SEMENTE)), {
        "atributos": opcoes_atributos,
        "modelo__learning_rate": [0.05, 0.1],
        "modelo__max_depth": [2, 3],
    }),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)  # 5 rodadas.
finais = {}  # Melhor pipeline de cada família.
linhas_val = []  # Resultados na validação.
for nome, (pipe, grade) in buscas.items():
    inicio = time.time()
    busca = GridSearchCV(pipe, grade, cv=cv, scoring="f1_macro", n_jobs=-1)
    busca.fit(X_tr, y_tr)  # Só o treino participa da escolha.
    segundos = time.time() - inicio
    finais[nome] = busca.best_estimator_  # Já reajustado com todo o treino.
    prev_val = busca.predict(X_val)  # Validação: compara famílias de modelos.
    usa_eng = busca.best_params_["atributos"] != "passthrough"
    params = {k.replace("modelo__", ""): v for k, v in busca.best_params_.items()
              if k != "atributos"}
    print(f"\n{nome}: melhores parâmetros {params} | engenharia de atributos: {usa_eng}")
    print(f"  F1 macro CV = {busca.best_score_:.4f} | F1 macro validação = "
          f"{f1_score(y_val, prev_val, average='macro'):.4f} | tempo da busca = {segundos:.0f} s")
    # Efeito isolado da engenharia de atributos: melhor resultado de CV com e sem ela.
    res = pd.DataFrame(busca.cv_results_)
    com = res[res.param_atributos.astype(str) != "passthrough"].mean_test_score.max()
    sem = res[res.param_atributos.astype(str) == "passthrough"].mean_test_score.max()
    print(f"  Melhor F1 CV sem novos atributos = {sem:.4f} | com novos atributos = {com:.4f}")
    linhas_val.append({"modelo": nome, "parametros": str(params),
                       "engenharia_atributos": usa_eng,
                       "f1_cv": busca.best_score_, "f1_cv_sem_eng": sem, "f1_cv_com_eng": com,
                       "f1_validacao": f1_score(y_val, prev_val, average="macro"),
                       "tempo_busca_s": segundos})
pd.DataFrame(linhas_val).round(4).to_csv(SAIDA / "modelos_validacao.csv", index=False)

# O SVC não devolve probabilidades. Para a curva ROC, calibramos o pipeline
# escolhido com validação cruzada interna (só no treino). As classes previstas
# continuam vindo do SVC original.
probabilistico = {nome: modelo for nome, modelo in finais.items()}
probabilistico["SVM"] = CalibratedClassifierCV(finais["SVM"], cv=5, ensemble=False).fit(X_tr, y_tr)

# ------------------------------------------------------------------
# 4. Avaliação final no teste (usado uma única vez)
# ------------------------------------------------------------------
classes = finais["SVM"].classes_  # Ordem alfabética das culturas.
y_te_bin = label_binarize(y_te, classes=classes)  # Uma coluna 0/1 por cultura.
linhas = []
for nome, modelo in finais.items():
    prev = modelo.predict(X_te)
    prob = probabilistico[nome].predict_proba(X_te)  # Probabilidade de cada cultura.
    linhas.append({
        "modelo": nome,
        "accuracy": accuracy_score(y_te, prev),
        "precision_macro": precision_score(y_te, prev, average="macro"),
        "recall_macro": recall_score(y_te, prev, average="macro"),
        "f1_macro": f1_score(y_te, prev, average="macro"),
        "roc_auc_macro_ovr": roc_auc_score(y_te, prob, multi_class="ovr", average="macro"),
        "erros": int((prev != y_te).sum()),
    })
    print(f"\n=== {nome}: relatório no teste ===")
    print(classification_report(y_te, prev, digits=3, zero_division=0))
    erros = pd.DataFrame({"real": y_te, "previsto": prev})[prev != y_te]
    print("Erros (real -> previsto):", list(zip(erros.real, erros.previsto)))

    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay(confusion_matrix(y_te, prev, labels=classes),
                           display_labels=classes).plot(ax=ax, cmap="Greens",
                                                        colorbar=False, xticks_rotation=90)
    ax.set_title(f"{nome}: matriz de confusão no teste (330 amostras)")
    fig.tight_layout()
    arquivo = nome.split()[0].lower().replace("á", "a")
    fig.savefig(SAIDA / f"matriz_confusao_{arquivo}.png", dpi=150)
    plt.close(fig)

metricas = pd.DataFrame(linhas)
print("\nComparação no teste:\n", metricas.round(4).to_string(index=False))
metricas.round(4).to_csv(SAIDA / "modelos_teste.csv", index=False)

# Curva ROC média (micro, um contra o resto) de cada modelo.
fig, ax = plt.subplots(figsize=(6, 5))
for nome in finais:
    prob = probabilistico[nome].predict_proba(X_te)
    fpr, tpr, _ = roc_curve(y_te_bin.ravel(), prob.ravel())  # Junta as 22 classes.
    auc = roc_auc_score(y_te_bin, prob, average="micro")
    ax.plot(fpr, tpr, label=f"{nome} (AUC micro = {auc:.4f})")
ax.plot([1e-4, 1], [1e-4, 1], "k--", linewidth=1)  # Classificador aleatório.
ax.set_xscale("log")  # Escala log para enxergar a região de poucos falsos positivos.
ax.set_xlim(1e-4, 1)
ax.set_xlabel("Taxa de falsos positivos (escala log)")
ax.set_ylabel("Taxa de verdadeiros positivos")
ax.set_title("Curvas ROC no teste (média micro, um contra o resto)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(SAIDA / "curvas_roc.png", dpi=160)
plt.close(fig)

# ------------------------------------------------------------------
# 5. Importância dos atributos
# ------------------------------------------------------------------
nomes_entrada = {}  # Nomes das colunas que chegam ao modelo (7 ou 11).
for nome, modelo in finais.items():
    passo = modelo.named_steps["atributos"]
    nomes_entrada[nome] = (list(criar_atributos(X_tr.head(1)).columns)
                           if passo != "passthrough" else ATRIBUTOS)

imp_tabelas = {}
for nome in ["Árvore de decisão", "Random Forest", "Gradient Boosting"]:
    m = finais[nome].named_steps["modelo"]
    imp_tabelas[nome] = pd.Series(m.feature_importances_, index=nomes_entrada[nome])
imp = pd.DataFrame(imp_tabelas).fillna(0).sort_values("Random Forest", ascending=False)
print("\nImportância por impureza (soma 1 em cada modelo):\n", imp.round(3))
imp.round(4).to_csv(SAIDA / "importancia_impureza.csv")

# Permutação no teste: vale para qualquer modelo, inclusive o SVM.
perm = {}
for nome, modelo in finais.items():
    r = permutation_importance(modelo, X_te, y_te, scoring="f1_macro",
                               n_repeats=20, random_state=SEMENTE, n_jobs=-1)
    perm[nome] = pd.Series(r.importances_mean, index=ATRIBUTOS)
perm = pd.DataFrame(perm).sort_values("Random Forest", ascending=False)
print("\nQueda do F1 macro ao embaralhar cada atributo original:\n", perm.round(3))
perm.round(4).to_csv(SAIDA / "importancia_permutacao.csv")

fig, ax = plt.subplots(figsize=(8, 4.5))
perm.iloc[::-1].plot.barh(ax=ax, width=0.8)
ax.set_xlabel("Queda do F1 macro no teste ao embaralhar o atributo")
ax.set_title("Importância por permutação")
fig.tight_layout()
fig.savefig(SAIDA / "importancia_permutacao.png", dpi=160)
plt.close(fig)

# Regras do topo da árvore (3 primeiros níveis), em unidades padronizadas.
arvore = finais["Árvore de decisão"].named_steps["modelo"]
print("\nÁrvore final | profundidade:", arvore.get_depth(), "| folhas:", arvore.get_n_leaves())
print(export_text(arvore, feature_names=nomes_entrada["Árvore de decisão"], max_depth=3))

# Árvore sem escala só para ler os cortes nas unidades originais.
leitura = DecisionTreeClassifier(**{k: v for k, v in arvore.get_params().items()})
leitura.fit(criar_atributos(X_tr) if nomes_entrada["Árvore de decisão"] != ATRIBUTOS else X_tr, y_tr)
print("Regras nas unidades originais (3 níveis):")
print(export_text(leitura, feature_names=nomes_entrada["Árvore de decisão"], max_depth=3, decimals=1))

# ------------------------------------------------------------------
# 6. Uso prático: recomendação para uma análise de solo nova
# ------------------------------------------------------------------
# O modelo final é escolhido pela validação, não pelo teste.
validacao = pd.DataFrame(linhas_val)
melhor = validacao.sort_values("f1_validacao", ascending=False, kind="stable").iloc[0].modelo
nova = pd.DataFrame([{"nitrogenio": 80, "fosforo": 45, "potassio": 40, "temperatura_c": 24.0,
                      "umidade_pct": 82.0, "ph": 6.3, "chuva_mm": 220.0}])
prob = pd.Series(probabilistico[melhor].predict_proba(nova)[0], index=classes).sort_values(ascending=False)
print(f"\nModelo usado: {melhor}. Três culturas mais prováveis para a área nova:")
print(prob.head(3).round(3))

# ------------------------------------------------------------------
# 7. Robustez: validação cruzada repetida (treino + validação, sem o teste)
# ------------------------------------------------------------------
# O teste tem 15 amostras por cultura; 1 erro muda a acurácia em 0,3 ponto.
# Repetir a validação cruzada mostra se as diferenças entre modelos são reais.
from sklearn.base import clone  # Cópia do pipeline sem o ajuste.
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
X_dev = pd.concat([X_tr, X_val])  # 1870 amostras (85%).
y_dev = pd.concat([y_tr, y_val])
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=SEMENTE)  # 20 rodadas.
robustez = []
for nome, modelo in finais.items():
    notas = cross_val_score(clone(modelo), X_dev, y_dev, cv=rcv, scoring="f1_macro", n_jobs=-1)
    robustez.append({"modelo": nome, "f1_medio": notas.mean(), "desvio": notas.std(),
                     "minimo": notas.min(), "maximo": notas.max()})
robustez = pd.DataFrame(robustez)
print("\nValidação cruzada repetida (5 x 4 = 20 rodadas, F1 macro):\n",
      robustez.round(4).to_string(index=False))
robustez.round(4).to_csv(SAIDA / "modelos_cv_repetida.csv", index=False)
