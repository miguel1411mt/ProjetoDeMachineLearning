"""Configurações compartilhadas pelos scripts do projeto."""

from pathlib import Path  # Caminhos de arquivos.
import pandas as pd  # Leitura da tabela.

# Pasta do projeto: a do arquivo, ou a pasta atual quando rodado no notebook.
PASTA = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
SAIDA = PASTA / "resultados"  # Tabelas e figuras geradas.
SAIDA.mkdir(exist_ok=True)  # Cria a pasta se ainda não existir.
SEMENTE = 42  # Semente fixa para todos os sorteios.

# Nomes das colunas em português, para facilitar a leitura do relatório.
COLUNAS = {"N": "nitrogenio", "P": "fosforo", "K": "potassio",
           "temperature": "temperatura_c", "humidity": "umidade_pct",
           "ph": "ph", "rainfall": "chuva_mm", "label": "cultura"}

# Tradução das 22 culturas do dataset.
CULTURAS = {"rice": "arroz", "maize": "milho", "chickpea": "grão-de-bico",
            "kidneybeans": "feijão-roxo", "pigeonpeas": "feijão-guandu",
            "mothbeans": "feijão-mariposa", "mungbean": "feijão-mungo",
            "blackgram": "feijão-preto-urd", "lentil": "lentilha",
            "pomegranate": "romã", "banana": "banana", "mango": "manga",
            "grapes": "uva", "watermelon": "melancia", "muskmelon": "melão",
            "apple": "maçã", "orange": "laranja", "papaya": "mamão",
            "coconut": "coco", "cotton": "algodão", "jute": "juta",
            "coffee": "café"}

ATRIBUTOS = ["nitrogenio", "fosforo", "potassio", "temperatura_c",
             "umidade_pct", "ph", "chuva_mm"]  # Sete entradas originais.


def carregar_dados():
    """Lê o CSV original do Kaggle e traduz colunas e culturas."""
    dados = pd.read_csv(PASTA / "dados" / "Crop_recommendation.csv")
    dados = dados.rename(columns=COLUNAS)  # Colunas em português.
    dados["cultura"] = dados["cultura"].map(CULTURAS)  # Culturas em português.
    return dados


def criar_atributos(tabela):
    """Engenharia de atributos: proporções entre N, P e K.

    O dataset descreve N, P e K como proporções no solo, então as razões
    entre eles podem carregar informação que os valores isolados não mostram.
    """
    nova = tabela.copy()  # Não altera a tabela recebida.
    nova["npk_total"] = nova.nitrogenio + nova.fosforo + nova.potassio
    nova["razao_n_p"] = nova.nitrogenio / (nova.fosforo + 1)  # +1 evita divisão por zero.
    nova["razao_n_k"] = nova.nitrogenio / (nova.potassio + 1)
    nova["razao_p_k"] = nova.fosforo / (nova.potassio + 1)
    return nova
