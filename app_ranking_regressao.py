import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sqlite3

# Função para inicializar o banco de dados e criar a tabela, se não existir
def init_db():
    conn = sqlite3.connect('metrics.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            name TEXT,
            class TEXT,
            mse REAL,
            mae REAL,
            rmse REAL,
            r2 REAL
        )
    ''')
    conn.commit()
    conn.close()

# Função para salvar as métricas no SQLite
def save_metrics(name, student_class, mse, mae, rmse, r2):
    conn = sqlite3.connect('metrics.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO metrics (name, class, mse, mae, rmse, r2)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (name, student_class, mse, mae, rmse, r2))
    conn.commit()
    conn.close()

# Função para obter o ranking dos alunos
def get_ranking():
    conn = sqlite3.connect('metrics.db')
    df = pd.read_sql_query('SELECT * FROM metrics', conn)
    df = df.sort_values(by=['rmse', 'mae', 'mse'], ascending=[True, True, True])
    df.reset_index(drop=True, inplace=True)  # Redefine os índices
    df.index += 1  # Começa os índices em 1
    conn.close()
    return df

# Função para obter o ranking por turma
def get_ranking_by_class(student_class):
    conn = sqlite3.connect('metrics.db')
    df = pd.read_sql_query('SELECT * FROM metrics WHERE class = ?', conn, params=(student_class,))
    df = df.sort_values(by=['rmse', 'mae', 'mse'], ascending=[True, True, True])
    df.reset_index(drop=True, inplace=True)  # Redefine os índices
    df.index += 1  # Começa os índices em 1
    conn.close()
    return df


# Inicializa o banco de dados
init_db()

st.set_page_config(layout="wide")
# Layout do Streamlit
st.title('Avaliação de Modelos de Alunos')

# Colunas para o formulário e ranking
col1, col2, col3 = st.columns([1, 0.5, 3])

with col1:
    # Leitura dos arquivos X_test e y_test
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv")

    # Entrada do nome do aluno e turma
    name = st.text_input("Nome do Aluno")
    student_class = st.selectbox("Selecione sua turma", ["ADS-SUPERIOR-M5", "TDS-SUBSEQUENTE-M3", "3ANO-A", "3ANO-B", "3ANO-C"])

    # Upload do arquivo CSV com os rótulos preditos
    uploaded_file = st.file_uploader("Faça upload do CSV com os rótulos preditos", type="csv")

    if uploaded_file is not None:
        y_pred = pd.read_csv(uploaded_file, index_col=0)

        # Garantir que y_pred e y_test tenham índices correspondentes
        y_test_filtered = y_test.loc[y_pred.index]

        # Calcular as métricas
        mse = mean_squared_error(y_test_filtered, y_pred)
        mae = mean_absolute_error(y_test_filtered, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_filtered, y_pred))
        r2 = r2_score(y_test_filtered, y_pred)

        st.write(f"MSE = {mse:.2f}")
        st.write(f"MAE = {mae:.2f}")
        st.write(f"RMSE = {rmse:.2f}")
        st.write(f"R2 = {r2:.2f}")

        # Botão para calcular e salvar o resultado
        if st.button("Cadastrar Resultado no Ranking"):
            save_metrics(name, student_class, mse, mae, rmse, r2)
            st.success("Resultado salvo com sucesso!")

with col2:
    pass
with col3:
    # Exibir o ranking geral
    st.subheader("Ranking Geral")
    ranking_df = get_ranking()
    st.write(ranking_df.head(5))

    # Exibir o ranking por turma
    st.subheader(f"Ranking por Turma ({student_class})")
   
    selected_class_ranking_df = get_ranking_by_class(student_class)
    st.write(selected_class_ranking_df)


