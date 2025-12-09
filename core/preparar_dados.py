# backend/core/preparar_dados.py
import pandas as pd
import numpy as np
import os  # <-- Importação adicionada
import sqlite3
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from .clima import is_frio # Importa a função existente is_frio

def carregar_dados_do_banco():
    """Carrega pedidos do banco e adiciona features contextuais."""
    # Caminho do banco
    DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'pizzaria.db') # <-- Corrigido
    conn = sqlite3.connect(DB_PATH)
    
    # Carrega dados brutos
    df = pd.read_sql_query("SELECT cliente_id, pizza, ingredientes, data_hora FROM pedidos", conn)
    conn.close()
    
    if df.empty:
        return pd.DataFrame()

    df['data_hora'] = pd.to_datetime(df['data_hora'])
    df['horario_pedido'] = df['data_hora'].dt.hour
    df['dia_semana'] = df['data_hora'].dt.dayofweek
    df['mes'] = df['data_hora'].dt.month
    # Adicione mais features contextuais aqui

    # Adiciona colunas booleanas de tipo de pizza (exemplo)
    df['eh_vegetariana'] = df['ingredientes'].str.contains('tomate|cebola|pimentão|ervilha|milho|berinjela|espinafre|brocolis|palmito', case=False, na=False)
    df['eh_picante'] = df['ingredientes'].str.contains('pepperoni|calabresa|pimenta|jalapeño|catupiry', case=False, na=False)
    df['eh_doce'] = df['pizza'].str.contains('Chocolate|Doce de Leite|Banana', case=False, na=False)

    # Adiciona clima baseado na função is_frio (simplificado)
    # Isso é uma aproximação, pois is_frio é baseado na data/hora atual, não no histórico.
    # Para fins de treinamento, você pode adicionar uma coluna 'esta_frio' baseada em uma heurística
    # ou usar o clima real se tiver um histórico.
    # Por enquanto, vamos adicionar uma coluna booleana 'esta_frio' baseada em uma heurística simples
    # ou deixar como um placeholder para treinamento e prever com a função real na hora da recomendação.
    # Vamos adicionar uma coluna placeholder que pode ser calculada dinamicamente.
    # df['esta_frio'] = is_frio() # <-- Isso não faz sentido para dados históricos
    # A coluna 'esta_frio' deve ser calculada no momento da previsão, não no treinamento.
    # Portanto, NÃO adicionamos 'clima' ou 'esta_frio' como uma coluna fixa aqui.

    return df

def preparar_features(df, cardapio_info):
    """Prepara features numéricas e categóricas para o modelo."""
    df_proc = df.copy()

    # 1. TF-IDF para ingredientes
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_proc['ingredientes'])
    # Converter para DataFrame para facilitar o merge
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'ingrediente_{i}' for i in range(tfidf_matrix.shape[1])])
    df_proc = pd.concat([df_proc.reset_index(drop=True), tfidf_df], axis=1)

    # 2. Codificar variáveis categóricas
    le_cliente = LabelEncoder()
    le_pizza = LabelEncoder()
    # le_clima = LabelEncoder() # Não mais necessário se clima for dinâmico

    df_proc['cliente_id_encoded'] = le_cliente.fit_transform(df_proc['cliente_id'])
    df_proc['pizza_encoded'] = le_pizza.fit_transform(df_proc['pizza'])
    # df_proc['clima_encoded'] = le_clima.fit_transform(df_proc['clima']) # Não mais necessário

    # 3. Normalizar features numéricas (opcional, mas recomendado para alguns modelos)
    scaler = StandardScaler()
    features_numericas = ['horario_pedido', 'dia_semana', 'mes'] # Adicione mais se necessário
    df_proc[features_numericas] = scaler.fit_transform(df_proc[features_numericas])

    # 4. Adicionar features do cardápio (ex: preço)
    # Supondo que cardapio_info seja um dict como {pizza_nome: {'preco': 30.0, ...}}
    df_proc['preco_pizza'] = df_proc['pizza'].map(lambda x: cardapio_info.get(x, {}).get('preco', 0))

    # Colunas que serão usadas como features para o modelo
    # Exclui colunas originais que já foram transformadas ou não são features
    feature_cols = [col for col in df_proc.columns if col not in ['cliente_id', 'pizza', 'ingredientes', 'data_hora']] # Removido 'clima' e 'clima_encoded' se existiam

    X = df_proc[feature_cols]
    y = df_proc['pizza_encoded'] # A pizza pedida (target)

    return X, y, vectorizer, le_cliente, le_pizza, scaler # Removido le_clima