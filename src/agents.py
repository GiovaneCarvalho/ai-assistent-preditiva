from typing import TypedDict, Optional
from typing import List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from config import llm
from retrievers import (
    retriever_manual_tecnico,
    retriever_perguntas_frequentes,
    retriever_politicas_procedimentos,
    retriever_tickets,
)


class State(TypedDict, total=False):
    query: str
    route: Optional[str]
    answer: Optional[str]
    chat_history: Optional[List[BaseMessage]]

def agent_with_retriever(
    state: State, papel: str, prompt_instrucoes: str, retriever=None
):
    """
    Cria um agente de IA para responder a uma query usando um retriever para buscar contexto.

    Esta função constrói uma cadeia de mensagens para o modelo de linguagem (LLM),
    incluindo o histórico do chat, o contexto recuperado por um retriever (se fornecido)
    e instruções específicas para o papel do agente. A resposta do LLM é então
    armazenada no estado e retornada.

    Args:
        state (State): O estado atual da conversa, contendo a query e o histórico.
        papel (str): O papel que o agente deve assumir (ex: "especialista em detalhes técnicos").
        prompt_instrucoes (str): Instruções detalhadas para o comportamento do agente.
        retriever (Optional): Um objeto retriever para buscar documentos relevantes
                               como contexto. Se `None`, nenhum contexto externo é usado.

    Returns:
        State: O estado da conversa atualizado com a resposta do LLM.
    """
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
    """
    Agente especializado em responder a perguntas sobre detalhes técnicos.

    Define um prompt específico para um especialista em suporte técnico e produto
    e utiliza o `retriever_manual_tecnico` para buscar informações.

    Args:
        state (State): O estado atual da conversa.

    Returns:
        State: O estado da conversa atualizado com a resposta do agente.
    """
    prompt_instrucoes = (
        "Seja um **especialista em suporte técnico e produto**. "
        "Você deve responder a perguntas sobre **especificações técnicas**, "
        "**instruções de instalação**, **manutenção preventiva** e **solução de problemas**. "
        "Sua resposta deve ser precisa, técnica e objetiva, baseada estritamente no manual técnico. "
        "Para problemas, ofereça uma solução clara e passo a passo."
    )
    return agent_with_retriever(
        state,
        "especialista em detalhes técnicos",
        prompt_instrucoes,
        retriever_manual_tecnico,
    )


def agent_perguntas_e_respostas(state: State):
    """
    Agente especializado em responder a Perguntas Frequentes (FAQ).

    Define um prompt para um agente focado em respostas diretas e concisas,
    usando o `retriever_perguntas_frequentes` para buscar informações na base de FAQ.

    Args:
        state (State): O estado atual da conversa.

    Returns:
        State: O estado da conversa atualizado com a resposta do agente.
    """
    prompt_instrucoes = (
        "Seja um **especialista em Perguntas Frequentes (FAQ)**. "
        "Sua função é fornecer respostas diretas e concisas a perguntas comuns. "
        "Responda como se estivesse consultando uma base de conhecimento, mantendo a resposta factual e sem rodeios. "
        "Se a pergunta se referir a um problema, ofereça a resposta e, se necessário, sugira o contato com o suporte técnico para casos complexos."
    )
    return agent_with_retriever(state, "especialista em FAQs", prompt_instrucoes, retriever_perguntas_frequentes)


def agent_politicas_e_procedimentos(state: State):
    """
    Agente especializado em responder a perguntas sobre políticas e procedimentos da empresa.

    Define um prompt formal para um agente que fornece informações sobre garantias,
    prazos e regras internas, utilizando o `retriever_politicas_procedimentos`
    para obter contexto.

    Args:
        state (State): O estado atual da conversa.

    Returns:
        State: O estado da conversa atualizado com a resposta do agente.
    """
    prompt_instrucoes = (
        "Seja um **especialista em políticas e procedimentos da empresa**. "
        "Sua tarefa é responder a perguntas sobre **garantia**, **horário de atendimento**, "
        "**prazos de SLA** e **regras internas de suporte**. "
        "Sua resposta deve ser formal e baseada nos documentos oficiais, garantindo que o cliente entenda as regras e os processos da empresa."
    )
    return agent_with_retriever(
        state,
        "especialista em políticas e procedimentos",
        prompt_instrucoes,
        retriever_politicas_procedimentos,
    )


def agent_tickets(state: State):
    """
    Agente especializado em fornecer informações sobre tickets de atendimento.

    Define um prompt direto para um agente focado em dados de tickets,
    usando o `retriever_tickets` para buscar status e detalhes de chamados existentes.

    Args:
        state (State): O estado atual da conversa.

    Returns:
        State: O estado da conversa atualizado com a resposta do agente.
    """
    prompt_instrucoes = (
        "Seja um **especialista em tickets de atendimento**. "
        "Você deve fornecer informações precisas sobre o **status e detalhes de um chamado existente**. "
        "Sua resposta deve ser direta, baseada nos dados do ticket (Ticket ID, Status, Responsável, Descrição do Problema). "
        "Se o usuário perguntar sobre um ticket específico, forneça as informações relevantes e mantenha a resposta curta e direta."
    )
    return agent_with_retriever(
        state,
        "especialista em tickets de atendimento",
        prompt_instrucoes,
        retriever_tickets,
    )