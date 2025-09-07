# supervisor.py

from langchain_core.messages import SystemMessage, HumanMessage
from config import llm
from agents import State

def supervisor(state: State):
    query = state["query"]
    chat_history = state.get("chat_history", [])

    mensagens = [
        SystemMessage(content=(
        """Você é um assistente virtual de atendimento ao cliente da **Industech**, uma empresa especializada em produtos industriais.

        Sua principal responsabilidade é atuar como um supervisor, **roteando as perguntas dos clientes para o agente especialista mais adequado**.

        Sua única função é analisar a pergunta do usuário e retornar **uma palavra-chave** que representa o agente responsável. Se a pergunta não se encaixar em nenhuma categoria de especialista, ou se for uma saudação ou uma pergunta sobre suas próprias capacidades, você deve responder de forma amigável diretamente ao cliente, sem rotear.

        A sua resposta deve ser:
        - **Uma frase de resposta direta**, caso a pergunta seja geral (ex: "Olá", "Tudo bem?", "O que você faz?").
        - **Uma das seguintes palavras-chave**, em minúsculas e sem pontuação, para roteamento:
        - **detalhe_tecnico**: para perguntas sobre **especificações técnicas**, **manuais de produtos** ou **solução de problemas**.
        - **perguntas_e_respostas**: para **dúvidas operacionais comuns** ou **FAQs**.
        - **politicas_e_procedimentos**: para questões sobre **políticas da empresa**, **garantia** ou **prazos de atendimento (SLA)**.
        - **tickets**: para perguntas sobre o **status de um chamado**.

        **Regras:**
        - Responda apenas com a frase ou com a palavra-chave.
        - Não adicione explicações, comentários ou qualquer outro texto.
        - Não invente novas categorias."""
        )),
        *chat_history,
        HumanMessage(content=query)
    ]
    
    resposta = llm.invoke(mensagens)
    resposta_limpa = resposta.content.strip().lower()

    # Verifica se a resposta é uma das rotas ou se é uma resposta direta
    if resposta_limpa in ["detalhe_tecnico", "perguntas_e_respostas", "politicas_e_procedimentos", "tickets"]:
        state["route"] = resposta_limpa
    else:
        state["answer"] = resposta.content # Salva a resposta direta na chave 'answer'

    return state