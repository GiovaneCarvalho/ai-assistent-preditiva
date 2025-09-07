import streamlit as st
from workflow import build_workflow
from langchain.schema import HumanMessage, AIMessage  # Importe as classes de mensagem

app = build_workflow()

st.set_page_config(page_title="Atendente Virtual", page_icon="🤖")
st.title("💬 Atendente Virtual")

# Inicialize o histórico de chat para a UI e para o backend
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # <-- Histórico para o LangChain

# Exibe as mensagens da UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Digite sua pergunta..."):
    # Adiciona a mensagem do usuário à UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepara o input para o workflow, incluindo o histórico
    workflow_input = {
        "query": prompt,
        "chat_history": st.session_state.chat_history,  # <-- PASSA O HISTÓRICO
    }

    # Invoca o workflow com o estado completo
    resultado = app.invoke(workflow_input)
    resposta = resultado["answer"]

    # Adiciona a resposta do assistente à UI
    st.session_state.messages.append({"role": "assistant", "content": resposta})
    with st.chat_message("assistant"):
        st.markdown(resposta)

    # ATUALIZA o histórico do backend para a próxima interação
    st.session_state.chat_history.extend(
        [HumanMessage(content=prompt), AIMessage(content=resposta)]
    )
