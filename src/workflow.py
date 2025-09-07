from langgraph.graph import StateGraph, START, END
from agents import (
    State,
    agent_detalhe_tecnico,
    agent_perguntas_e_respostas,
    agent_politicas_e_procedimentos,
    agent_tickets,
)
from supervisor import supervisor

# Função para decidir o próximo passo
def decide_action(state: State):
    # Se a resposta já foi gerada pelo supervisor, encerra o fluxo.
    if "answer" in state:
        return "end_workflow" 
    else:
        # Caso contrário, usa a rota para ir para o agente especialista.
        return state["route"]

def build_workflow():
    workflow = StateGraph(State)

    # Adicione os nós (nodes)
    workflow.add_node("supervisor_node", supervisor)
    workflow.add_node("detalhe_tecnico_node", agent_detalhe_tecnico)
    workflow.add_node("perguntas_e_respostas_node", agent_perguntas_e_respostas)
    workflow.add_node("politicas_e_procedimentos_node", agent_politicas_e_procedimentos)
    workflow.add_node("tickets_node", agent_tickets)
    
    # Adicione um nó de saída para a resposta direta
    workflow.add_node("end_workflow", lambda x: x)

    # Defina o ponto de entrada
    workflow.add_edge(START, "supervisor_node")

    # Adicione o roteamento condicional
    workflow.add_conditional_edges(
        "supervisor_node",
        decide_action,
        {
            "detalhe_tecnico": "detalhe_tecnico_node",
            "perguntas_e_respostas": "perguntas_e_respostas_node",
            "politicas_e_procedimentos": "politicas_e_procedimentos_node",
            "tickets": "tickets_node",
            "end_workflow": END # Termina o fluxo se a resposta já foi gerada pelo supervisor
        }
    )

    # Defina as saídas dos agentes especialistas
    workflow.add_edge("detalhe_tecnico_node", END)
    workflow.add_edge("perguntas_e_respostas_node", END)
    workflow.add_edge("politicas_e_procedimentos_node", END)
    workflow.add_edge("tickets_node", END)

    return workflow.compile()