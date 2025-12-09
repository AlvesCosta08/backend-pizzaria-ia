# Sistema de Recomendação de Pizzas - Backend

Este projeto implementa um backend para uma pizzaria com funcionalidades de **recomendação de pizzas personalizada**. Ele utiliza técnicas de **processamento de linguagem natural (TF-IDF)** e **aprendizado de máquina (Random Forest)** para sugerir a pizza ideal com base no histórico do cliente, contexto (horário, dia da semana, clima) e preferências alimentares (vegetariano).

## 🚀 Funcionalidades

*   **Recomendação Simples (`/api/recomendar`)**: Utiliza regras baseadas em histórico de pedidos, horários populares, perfis semelhantes e clima para sugerir uma pizza.
*   **Recomendação Avançada (`/api/recomendar_avancado`)**: Emprega um modelo de **Random Forest** treinado com dados históricos para prever a pizza mais provável para um cliente em um determinado contexto.
*   **CRUD de Pedidos (`/api/pedidos`)**: Permite criar, ler, atualizar e deletar registros de pedidos no banco de dados.
*   **Persistência de Dados**: Utiliza **SQLite** para armazenar os pedidos e informações dos clientes.
*   **Modelos Treinados**: Modelos de IA são salvos e carregados da pasta `core/models/` para persistência entre execuções.
*   **API RESTful**: Fornece endpoints HTTP padronizados para integração com frontends ou outros serviços.

## 🛠️ Tecnologias Utilizadas

*   **Python 3.11**: Linguagem de programação principal.
*   **Flask**: Framework web para criação da API.
*   **SQLite**: Banco de dados relacional leve.
*   **Pandas**: Manipulação e análise de dados.
*   **NumPy**: Computação científica.
*   **Scikit-learn**: Biblioteca para aprendizado de máquina (Random Forest, TF-IDF, LabelEncoder, StandardScaler).
*   **Joblib**: Serialização e persistência de modelos de aprendizado de máquina.
*   **Flask-CORS**: Habilita o compartilhamento de recursos de origem cruzada (CORS).

## 📁 Estrutura do Projeto

```
BACKEND/
├── .venv/ # (Opcional) Ambiente virtual
├── core/
│ ├── init.py
│ ├── models/ # Pasta para modelos de IA treinados (modelo_recomendacao.pkl)
│ ├── cardapio.py # Definição do cardápio e extras
│ ├── clima.py # Função para simular clima frio
│ ├── modelo_avancado.py # Lógica do modelo de recomendação avançado
│ ├── pizza_recommender.py # Lógica da recomendação simples e CRUD de pedidos
│ └── preparar_dados.py # Funções para carregar e preparar dados para o modelo
├── data/
│ ├── pedidos.csv # (Exemplo) Dados históricos de pedidos
│ └── pizzaria.db # Banco de dados SQLite gerado
├── app.py # Ponto de entrada da aplicação Flask
├── config.py # Configurações da aplicação
├── Dockerfile # Definição para containerização (estágio produtivo)
├── requirements.txt # Dependências do projeto
└── wsgi.py # Ponto de entrada WSGI (para servidores como Gunicorn)

```

## 📋 Pré-requisitos

*   Python 3.11 ou superior
*   Pip (gerenciador de pacotes do Python)

## 🔧 Instalação e Execução

1.  **Clone o repositório** (ou crie a estrutura manualmente):
    ```bash
    git clone <url_do_seu_repositorio>
    cd BACKEND
    ```

2.  **Crie e ative um ambiente virtual (opcional, mas recomendado)**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # No Linux/Mac
    # .venv\Scripts\activate   # No Windows
    ```

3.  **Instale as dependências**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute a aplicação**:
    ```bash
    python app.py
    ```
    A API estará disponível em `http://localhost:5000`.

## 📡 Endpoints da API

*   **`GET /api/recomendar?cliente_id=<int>`**: Recomenda uma pizza com base em regras simples.
*   **`GET /api/recomendar_avancado?cliente_id=<int>`**: Recomenda uma pizza usando o modelo avançado de IA. **Requer que o modelo tenha sido treinado previamente.**
*   **`POST /api/pedido`**: Salva um novo pedido no banco de dados.
*   **`GET /api/pedidos`**: Lista todos os pedidos.
*   **`GET /api/pedido/<id>`**: Obtém um pedido específico.
*   **`PUT /api/pedido/<id>`**: Atualiza um pedido específico.
*   **`DELETE /api/pedido/<id>`**: Deleta um pedido específico.

## 🧠 Funcionamento do Modelo de IA

O modelo avançado (`modelo_avancado.py`) funciona da seguinte maneira:

1.  **Carrega Dados**: Lê os pedidos antigos do banco de dados `pizzaria.db`.
2.  **Prepara Features**: Extrai características dos pedidos, como:
    *   ID do cliente (codificado).
    *   Horário, dia da semana e mês do pedido.
    *   Preço da pizza.
    *   Tipo de pizza (vegetariana, picante, doce).
    *   Ingredientes (usando TF-IDF).
    *   Clima (se está frio ou não no momento da previsão).
3.  **Treina o Modelo**: Utiliza um classificador `RandomForestClassifier` para aprender a mapear essas features para o nome da pizza pedida.
4.  **Salva o Modelo**: O modelo treinado, junto com os encoders e vectorizers, é salvo em `core/models/modelo_recomendacao.pkl`.
5.  **Faz Previsões**: Quando solicitado, o modelo carrega o `.pkl` salvo, processa o contexto atual e as pizzas do cardápio, e retorna a pizza com maior probabilidade de ser pedida.

## 🐳 Docker (Opcional)

O projeto inclui um `Dockerfile` para containerização em ambiente produtivo.

1.  **Construa a imagem**:
    ```bash
    docker build -t backend-pizza .
    ```
2.  **Execute o contêiner**:
    ```bash
    docker run -p 8000:8000 backend-pizza
    ```
    A API estará disponível em `http://localhost:8000`.

> **Dica**: Para persistir o banco de dados e os modelos treinados, utilize volumes Docker ao executar o contêiner.

## 📝 Licença

Este projeto é de código aberto e está disponível sob a Licença MIT.
# backend-pizzaria-ia
