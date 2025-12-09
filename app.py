# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from config import Config
from core.pizza_recommender import init_db, bp as pizza_bp, bp_pedidos
from datetime import datetime
from core.modelo_avancado import inicializar_modelo_ia, modelo_ia
from core.clima import is_frio
from core.cardapio import CARDAPIO
import os

# Garante que a pasta 'models' exista
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'core', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app)

    init_db(app)
    inicializar_modelo_ia(app)

    @app.route('/api/recomendar_avancado', methods=['GET'])
    def recomendar_avancado():
        cliente_id = request.args.get('cliente_id', type=int)
        if not cliente_id:
            return jsonify({"erro": "cliente_id é obrigatório"}), 400

        agora = datetime.now()
        horario = agora.hour
        dia_semana = agora.weekday()
        mes = agora.month
        esta_frio_valor = is_frio()

        pizza, motivo = modelo_ia.prever_pizza(cliente_id, horario, dia_semana, mes, esta_frio_valor)

        if pizza:
            ingredientes = CARDAPIO.get(pizza, CARDAPIO["Margherita"])["ingredientes"]
            return jsonify({
                "pizza_recomendada": pizza,
                "ingredientes": [ing.strip() for ing in ingredientes.split(",")],
                "motivo": motivo
            })
        else:
            # Fallback para o modelo antigo
            print(f"Falha no modelo avançado: {motivo}. Usando modelo antigo.")
            return recomendar_fallback(cliente_id)

    def recomendar_fallback(cliente_id):
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
            return jsonify({"erro": "Falha na recomendação do fallback", "detalhe": str(e)}), 500

    def encontrar_clientes_similares(cliente_id_alvo: int, df: pd.DataFrame):
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

    app.register_blueprint(pizza_bp, url_prefix='/api')
    app.register_blueprint(bp_pedidos, url_prefix='/api')

    return app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)