from typing import TypedDict, Optional
from langchain.schema import SystemMessage, HumanMessage
from config import llm
from retrievers import (
    retriever_manual_tecnico,
    retriever_perguntas_frequentes,
    retriever_procedimentos,
    retriever_tickets,
)
from typing import List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage


class State(TypedDict, total=False):
    query: str
    route: Optional[str]
    answer: Optional[str]
    chat_history: Optional[List[BaseMessage]]


def agent_with_retriever(state: State, papel: str, prompt_instrucoes: str, retriever=None):
    query = state["query"]
    chat_history = state.get("chat_history", [])
    contexto = ""

    if retriever:
        recuperados = retriever.get_relevant_documents(query)
        if recuperados:
            contexto = "\n\n".join([d.page_content for d in recuperados])

    mensagens = [
        SystemMessage(
            content=(
                f"Você é um {papel}. "
                f"Suas instruções:\n{prompt_instrucoes}\n\n"
                f"- Use sempre o contexto recuperado para responder à ÚLTIMA pergunta do usuário.\n"
                f"- Use o histórico da conversa para entender o contexto geral e perguntas de acompanhamento.\n"
                f"- Se não houver informações relevantes no contexto, diga que não encontrou dados suficientes para responder.\n"
                f"- Evite inventar informações."
            )
        ),
        *chat_history,
        HumanMessage(
            content=(
                f"Pergunta do usuário:\n{query}\n\n"
                f"Contexto disponível para esta pergunta:\n{contexto if contexto else '[Nenhum contexto encontrado]'}"
            )
        ),
    ]

    resposta = llm.invoke(mensagens)
    state["answer"] = resposta.content
    return state


def agent_detalhe_tecnico(state: State):
    prompt_instrucoes = (
        "Seja um **especialista em suporte técnico e produto**. "
        "Você deve responder a perguntas sobre **especificações técnicas**, "
        "**instruções de instalação**, **manutenção preventiva** e **solução de problemas**. "
        "Sua resposta deve ser precisa, técnica e objetiva, baseada estritamente no manual técnico. "
        "Para problemas, ofereça uma solução clara e passo a passo."
    )
    return agent_with_retriever(
        state, "especialista em detalhes técnicos", prompt_instrucoes, retriever_manual_tecnico
    )


def agent_perguntas_e_respostas(state: State):
    prompt_instrucoes = (
        "Seja um **especialista em Perguntas Frequentes (FAQ)**. "
        "Sua função é fornecer respostas diretas e concisas a perguntas comuns. "
        "Responda como se estivesse consultando uma base de conhecimento, mantendo a resposta factual e sem rodeios. "
        "Se a pergunta se referir a um problema, ofereça a resposta e, se necessário, sugira o contato com o suporte técnico para casos complexos."
    )
    return agent_with_retriever(
        state, "especialista em FAQs", prompt_instrucoes, retriever_perguntas_frequentes
    )


def agent_politicas_e_procedimentos(state: State):
    prompt_instrucoes = (
        "Seja um **especialista em políticas e procedimentos da empresa**. "
        "Sua tarefa é responder a perguntas sobre **garantia**, **horário de atendimento**, "
        "**prazos de SLA** e **regras internas de suporte**. "
        "Sua resposta deve ser formal e baseada nos documentos oficiais, garantindo que o cliente entenda as regras e os processos da empresa."
    )
    return agent_with_retriever(
        state, "especialista em políticas e procedimentos", prompt_instrucoes, retriever_procedimentos
    )


def agent_tickets(state: State):
    prompt_instrucoes = (
        "Seja um **especialista em tickets de atendimento**. "
        "Você deve fornecer informações precisas sobre o **status e detalhes de um chamado existente**. "
        "Sua resposta deve ser direta, baseada nos dados do ticket (Ticket ID, Status, Responsável, Descrição do Problema). "
        "Se o usuário perguntar sobre um ticket específico, forneça as informações relevantes e mantenha a resposta curta e direta."
    )
    return agent_with_retriever(
        state, "especialista em tickets de atendimento", prompt_instrucoes, retriever_tickets
    )