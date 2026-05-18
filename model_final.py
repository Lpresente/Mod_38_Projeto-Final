#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np 
import pickle
import os

# Configuração inicial da página
st.set_page_config(page_title="Credit Scoring App", layout="wide", initial_sidebar_state="expanded")

st.title("🚀 Sistema de Escoragem de Crédito")
st.markdown("""
Este aplicativo utiliza um modelo de Regressão Logística pré-treinado para prever a probabilidade de inadimplência 
de novos clientes com base em um arquivo de dados (.csv).
""")

# Caminho do modelo salvo no módulo anterior
MODEL_PATH = 'model_final.pkl'

# Função para carregar o modelo treinado (armazenada em cache para não reprocessar a cada clique)
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        st.error(f"Erro: O arquivo '{MODEL_PATH}' não foi encontrado no diretório atual.")
        return None

model_pipeline = load_model()

# Sidebar para upload do arquivo
st.sidebar.header("Upload de Dados")
uploaded_file = st.sidebar.file_uploader("Selecione o arquivo CSV com os clientes:", type=["csv"])

if uploaded_file is not None and model_pipeline is not None:
    try:
        # 1. Carregar o CSV enviado pelo usuário
        df_input = pd.read_csv(uploaded_file)
        
        st.subheader("📊 Visualização dos Dados Carregados")
        st.dataframe(df_input.head(10))
        st.info(f"O arquivo contém {df_input.shape[0]} linhas e {df_input.shape[1]} colunas.")
        
        # Botão para disparar a escoragem
        if st.button("Executar Escoragem de Crédito"):
            st.write("---")
            with st.spinner("Processando dados e calculando scores..."):
                
                # Criar uma cópia para trabalhar os resultados
                df_output = df_input.copy()
                
                # O pipeline do Scikit-Learn precisa das features idênticas ao treino. 
                # Vamos remover as colunas 'index', 'data_ref' ou 'mau' caso existam no CSV enviado.
                vars_remover = ['data_ref', 'index', 'mau']
                features_escoragem = [col for col in df_input.columns if col not in vars_remover]
                
                # 2. Pipeline de Pré-processamento e Escoragem (Embutidos no model_pipeline)
                X_input = df_input[features_escoragem]
                
                # Calcular as probabilidades (A segunda coluna [:, 1] indica a probabilidade de ser 'Mau')
                probabilidades = model_pipeline.predict_proba(X_input)[:, 1]
                
                # Definir as predições finais usando o limiar padrão (0.5)
                predicoes = model_pipeline.predict(X_input)
                
                # 3. Adicionar os resultados no dataframe de saída
                df_output['Probabilidade_Inadimplencia'] = probabilidades
                df_output['Classificacao_Risco'] = np.where(predicoes == 1, 'Alto Risco (Mau)', 'Baixo Risco (Bom)')
                
                # Mostrar os resultados na tela
                st.subheader("🎯 Resultados da Escoragem")
                
                # Destacar visualmente quem é Alto Risco
                st.dataframe(df_output[['index', 'renda', 'tempo_emprego', 'Probabilidade_Inadimplencia', 'Classificacao_Risco']].head(20))
                
                # 4. Criar botão para download do arquivo completo escorado
                csv_download = df_output.to_csv(index=False).encode('utf-8')
                
                st.sidebar.markdown("---")
                st.sidebar.download_button(
                    label="📥 Baixar Base Escorada (CSV)",
                    data=csv_download,
                    file_name="clientes_escorados.csv",
                    mime="text/csv"
                )
                st.success("Escoragem concluída com sucesso! O botão para baixar o relatório está disponível no menu lateral.")
                
    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo: {e}")
        st.warning("Verifique se o CSV carregado possui os mesmos nomes de colunas que a base original de treino.")

else:
    if model_pipeline is not None:
        st.warning("Aguardando o upload de um arquivo CSV no menu lateral para iniciar.")


# In[ ]:




