



# 🚀 Projeto Credit Scoring: Da Modelagem à Produção com Streamlit

Este repositório contém o desenvolvimento completo de um modelo de **Credit Scoring** para previsão de inadimplência de clientes. O projeto engloba desde a análise exploratória de dados e treinamento de um modelo preditivo robusto (Regressão Logística) até a construção de uma aplicação web interativa utilizando o **Streamlit** para escoragem automática em ambiente de produção.

---

## 💻 Demonstração do Aplicativo

Aqui está o funcionamento prático da ferramenta de escoragem em funcionamento:

<img width="800" height="450" alt="Streamlit_teste" src="https://github.com/user-attachments/assets/783e1c57-d3bb-4ca4-b62e-0ec02ba5383d" />


## 📂 Estrutura do Repositório

O projeto está organizado da seguinte forma:

```text
├── Mod38Exercicio1.ipynb     # Análise univariada, bivariada e amostragem temporal (OOT)
├── Mod38Projeto.ipynb        # Modelagem completa e exportação do artefato final
├── app.py                    # Código-fonte da aplicação interativa Streamlit
├── model_final.pkl           # Pipeline e modelo de Regressão Logística treinados (Pickle)
├── credit_scoring.ftr        # Base de dados original no formato Apache Arrow (Feather)
└── base_teste_escoragem.csv  # Base limpa gerada para testes de escoragem no app
