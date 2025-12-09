# core/pizza_recommender.py
import pandas as pd
import numpy as np
import os
from datetime import datetime
from collections import Counter
from flask import Blueprint, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .cardapio import CARDAPIO, EXTRAS
from .clima import is_frio
import sqlite3
from contextlib import contextmanager

bp = Blueprint('pizza', __name__)
bp_pedidos = Blueprint('pedidos', __name__)

# Caminho do banco de dados
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'pizzaria.db')

@contextmanager
def get_db_connection():
    """Context manager para conexão com o banco de dados."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Retorna dicts
    try:
        yield conn
    finally:
        conn.close()

def init_db(app):
    """Inicializa o banco de dados e cria tabelas se não existirem, garantindo colunas necessárias."""
    with app.app_context():
        with get_db_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS pedidos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cliente_id INTEGER NOT NULL,
                    nome_cliente TEXT DEFAULT "",
                    telefone_cliente TEXT DEFAULT "",
                    data_hora TEXT NOT NULL,
                    pizza TEXT NOT NULL,
                    ingredientes TEXT NOT NULL,
                    valor REAL NOT NULL
                )
            ''')

            cursor = conn.execute("PRAGMA table_info(pedidos);")
            colunas_existentes = [row[1] for row in cursor.fetchall()]

            if 'nome_cliente' not in colunas_existentes:
                print("Adicionando coluna 'nome_cliente' à tabela 'pedidos'.")
                conn.execute('ALTER TABLE pedidos ADD COLUMN nome_cliente TEXT DEFAULT "";')

            if 'telefone_cliente' not in colunas_existentes:
                print("Adicionando coluna 'telefone_cliente' à tabela 'pedidos'.")
                conn.execute('ALTER TABLE pedidos ADD COLUMN telefone_cliente TEXT DEFAULT "";')

            conn.commit()
            print("Banco de dados inicializado ou verificado com sucesso.")

def is_vegetariano(cliente_id: int) -> bool:
    carnes = {'pepperoni', 'calabresa', 'frango', 'presunto', 'bacon', 'carne', 'salsicha'}
    try:
        with get_db_connection() as conn:
            cursor = conn.execute('SELECT ingredientes FROM pedidos WHERE cliente_id = ?', (cliente_id,))
            historico = cursor.fetchall()
        for row in historico:
            ingredientes = set(ing.strip().lower() for ing in row['ingredientes'].split(','))
            if ingredientes & carnes:
                return False
        return True
    except:
        return False

@bp.route('/recomendar', methods=['GET'])
def recomendar():
    cliente_id = request.args.get('cliente_id', type=int)
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM pedidos", conn)

        if df.empty:
            pizza = list(CARDAPIO.keys())[0]
            return jsonify({
                "pizza_recomendada": pizza,
                "ingredientes": [ing.strip() for ing in CARDAPIO[pizza]["ingredientes"].split(",")],
                "motivo": "Nenhum pedido registrado. Nossa sugestão especial!"
            })

        df['data_hora'] = pd.to_datetime(df['data_hora'], errors='coerce')
        df = df.dropna(subset=['data_hora'])
        df['dia_semana'] = df['data_hora'].dt.day_name()
        df['hora'] = df['data_hora'].dt.hour

        agora = datetime.now()
        dia_atual = agora.strftime('%A')
        hora_atual = agora.hour

        if cliente_id and cliente_id in df['cliente_id'].values:
            clientes_similares = encontrar_clientes_similares(cliente_id, df)
            df_contexto = df[
                (df['dia_semana'] == dia_atual) &
                (df['hora'] >= max(0, hora_atual - 2)) &
                (df['hora'] <= min(23, hora_atual + 2))
            ]
            if not clientes_similares.empty:
                pedidos_similares = df_contexto[df_contexto['cliente_id'].isin(clientes_similares['cliente_id'])]
                if not pedidos_similares.empty:
                    pizza = Counter(pedidos_similares['pizza']).most_common(1)[0][0]
                    motivo = f"Clientes com gostos parecidos com você pediram isso!"
                else:
                    historico_cliente = df[df['cliente_id'] == cliente_id]['pizza']
                    pizza = Counter(historico_cliente).most_common(1)[0][0]
                    motivo = f"Baseado no seu histórico. Você pediu essa {len(historico_cliente)}x!"
            else:
                historico_cliente = df[df['cliente_id'] == cliente_id]['pizza']
                pizza = Counter(historico_cliente).most_common(1)[0][0] if not historico_cliente.empty else list(CARDAPIO.keys())[0]
                motivo = f"Você sempre pede essa! ({len(historico_cliente)}x)" if not historico_cliente.empty else "Nossa sugestão especial!"
        else:
            df_contexto = df[
                (df['dia_semana'] == dia_atual) &
                (df['hora'] >= max(0, hora_atual - 2)) &
                (df['hora'] <= min(23, hora_atual + 2))
            ]
            if not df_contexto.empty:
                pizza = Counter(df_contexto['pizza']).most_common(1)[0][0]
                motivo = f"Popular hoje ({dia_atual}) neste horário! 🕒"
            else:
                pizza = list(CARDAPIO.keys())[0]
                motivo = "Nossa sugestão especial!"

        df_candidatas = df.copy()
        if cliente_id and is_vegetariano(cliente_id):
            df_candidatas = df_candidatas[
                ~df_candidatas['ingredientes'].str.contains(
                    r'pepperoni|calabresa|frango|presunto|bacon|carne|salsicha',
                    case=False, na=False, regex=True
                )
            ]
            if df_candidatas[df_candidatas['pizza'] == pizza].empty:
                 pizza_valida = df_candidatas['pizza'].iloc[0] if not df_candidatas.empty else pizza
                 pizza = pizza_valida
                 motivo = "Recomendação adaptada ao seu perfil vegetariano."

        if is_frio():
            pizzas_quentes = {"Calabresa", "Pepperoni", "Frango com Catupiry", "Quatro Queijos"}
            if pizza not in pizzas_quentes:
                pizza_quente = df_candidatas[df_candidatas['pizza'].isin(pizzas_quentes)]
                if not pizza_quente.empty:
                    pizza = pizza_quente['pizza'].iloc[0]
                    motivo += " E está frio! Sugerimos algo quente."

        ingredientes = CARDAPIO.get(pizza, CARDAPIO["Margherita"])["ingredientes"]
        return jsonify({
            "pizza_recomendada": pizza,
            "ingredientes": [ing.strip() for ing in ingredientes.split(",")],
            "motivo": motivo
        })
    except Exception as e:
        return jsonify({"erro": "Falha na recomendação", "detalhe": str(e)}), 500

def encontrar_clientes_similares(cliente_id_alvo: int, df: pd.DataFrame):
    """Encontra clientes com gostos alimentares semelhantes."""
    try:
        perfil_clientes = df.groupby('cliente_id')['ingredientes'].apply(lambda x: ' '.join(x)).reset_index()
        perfil_clientes['ingredientes'] = perfil_clientes['ingredientes'].str.lower()

        if cliente_id_alvo not in perfil_clientes['cliente_id'].values:
            return pd.DataFrame()

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(perfil_clientes['ingredientes'])

        cliente_idx = perfil_clientes[perfil_clientes['cliente_id'] == cliente_id_alvo].index[0]
        similarities = cosine_similarity(tfidf_matrix[cliente_idx], tfidf_matrix).flatten()

        perfil_clientes['similaridade'] = similarities
        similares = perfil_clientes[perfil_clientes['cliente_id'] != cliente_id_alvo].sort_values(by='similaridade', ascending=False)
        return similares.head(3)
    except:
        return pd.DataFrame()

@bp.route('/pedido', methods=['POST'])
def salvar_pedido():
    data = request.json
    cliente_id = data['cliente_id']
    nome_cliente = data.get('nome_cliente', '')
    telefone_cliente = data.get('telefone_cliente', '')
    pizza_nome = data['pizza']
    extras = data.get('extras', [])

    base = CARDAPIO.get(pizza_nome)
    if not base:
        return jsonify({"sucesso": False, "erro": "Pizza não encontrada"}), 400

    ingredientes = base["ingredientes"]
    valor = base["preco"]

    if extras:
        for extra in extras:
            if extra in EXTRAS:
                valor += EXTRAS[extra]
                ingredientes += f", {extra}"

    try:
        with get_db_connection() as conn:
            conn.execute('''
                INSERT INTO pedidos (cliente_id, nome_cliente, telefone_cliente, data_hora, pizza, ingredientes, valor)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (cliente_id, nome_cliente, telefone_cliente, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pizza_nome, ingredientes, round(valor, 2)))
            conn.commit()
        return jsonify({"sucesso": True, "mensagem": f"Pedido confirmado! Total: R$ {valor:.2f}"})
    except Exception as e:
        return jsonify({"sucesso": False, "erro": str(e)}), 500

# --- ROTAS DE PEDIDOS (CRUD) ---
@bp_pedidos.route('/pedidos', methods=['GET'])
def listar_pedidos():
    try:
        with get_db_connection() as conn:
            query = '''
                SELECT id, cliente_id, nome_cliente, telefone_cliente, data_hora, pizza, ingredientes, valor
                FROM pedidos
                ORDER BY data_hora DESC
            '''
            rows = conn.execute(query).fetchall()
            pedidos_list = [dict(row) for row in rows]
        return jsonify(pedidos_list), 200
    except Exception as e:
        return jsonify({"erro": "Falha ao carregar pedidos", "detalhe": str(e)}), 500

@bp_pedidos.route('/pedido/<int:id>', methods=['GET'])
def obter_pedido(id: int):
    try:
        with get_db_connection() as conn:
            row = conn.execute('SELECT * FROM pedidos WHERE id = ?', (id,)).fetchone()
            if row:
                return jsonify(dict(row)), 200
            else:
                return jsonify({"erro": "Pedido não encontrado"}), 404
    except Exception as e:
        return jsonify({"erro": "Falha ao obter pedido", "detalhe": str(e)}), 500

@bp_pedidos.route('/pedido/<int:id>', methods=['PUT', 'PATCH'])
def atualizar_pedido(id: int):
    dados = request.json
    nome_cliente = dados.get('nome_cliente')
    telefone_cliente = dados.get('telefone_cliente')
    pizza = dados.get('pizza')
    ingredientes = dados.get('ingredientes')
    valor = dados.get('valor')

    if nome_cliente is None and telefone_cliente is None and pizza is None and ingredientes is None and valor is None:
        return jsonify({"sucesso": False, "erro": "Nenhum campo para atualizar foi fornecido"}), 400

    try:
        with get_db_connection() as conn:
            updates = []
            params = []
            if nome_cliente is not None:
                updates.append("nome_cliente = ?")
                params.append(nome_cliente)
            if telefone_cliente is not None:
                updates.append("telefone_cliente = ?")
                params.append(telefone_cliente)
            if pizza is not None:
                updates.append("pizza = ?")
                params.append(pizza)
            if ingredientes is not None:
                updates.append("ingredientes = ?")
                params.append(ingredientes)
            if valor is not None:
                updates.append("valor = ?")
                params.append(valor)

            if not updates:
                return jsonify({"sucesso": False, "erro": "Nenhum campo válido para atualizar"}), 400

            query = f"UPDATE pedidos SET {', '.join(updates)} WHERE id = ?"
            params.append(id)

            cursor = conn.execute(query, params)  # <-- Correção aqui
            conn.commit()

            if cursor.rowcount == 0:  # <-- Correção aqui
                return jsonify({"sucesso": False, "erro": "Pedido não encontrado"}), 404

        return jsonify({"sucesso": True, "mensagem": "Pedido atualizado com sucesso!"}), 200
    except Exception as e:
        return jsonify({"sucesso": False, "erro": str(e)}), 500

@bp_pedidos.route('/pedido/<int:id>', methods=['DELETE'])
def deletar_pedido(id: int):
    try:
        with get_db_connection() as conn:
            cursor = conn.execute('DELETE FROM pedidos WHERE id = ?', (id,))
            conn.commit()
            if cursor.rowcount == 0:
                return jsonify({"sucesso": False, "erro": "Pedido não encontrado"}), 404
        return jsonify({"sucesso": True, "mensagem": "Pedido excluído com sucesso!"}), 200
    except Exception as e:
        return jsonify({"sucesso": False, "erro": str(e)}), 500