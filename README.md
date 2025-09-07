# Atendente Virtual - InduTech 🤖

Este projeto é um **Atendente Virtual** desenvolvido para a empresa fictícia **InduTech**, especializada em produtos industriais. O assistente foi construído utilizando a biblioteca **LangChain** e o framework **LangGraph**, que permite a criação de fluxos de trabalho com múltiplos agentes especializados. A interface do usuário é uma aplicação web simples, criada com **Streamlit**.

## 🧠 Como Funciona

O sistema opera com uma arquitetura de múltiplos agentes, orquestrada por um supervisor. O fluxo de trabalho (workflow) segue estes passos:

1.  **Entrada do Usuário**: O usuário envia uma pergunta através da interface do Streamlit. A `query` e o histórico de chat (`chat_history`) são encapsulados em um objeto de `state`.

2.  **Roteamento por Supervisor**: A pergunta é primeiro direcionada para um agente **supervisor**. Este agente analisa a intenção do usuário e decide qual agente especialista é o mais adequado para responder. Ele faz isso roteando a pergunta para uma de quatro categorias, ou responde diretamente se a pergunta for uma saudação ou algo geral.

3.  **Ação do Agente Especialista**: Com base na decisão do supervisor, a pergunta é enviada para o agente especialista correspondente. Cada agente é configurado com um `prompt` específico e um `retriever` para buscar informações em sua base de conhecimento dedicada. Os agentes são:

      - `agent_detalhe_tecnico`: Para questões sobre **especificações técnicas** e **manuais de produtos**.
      - `agent_perguntas_e_respostas`: Para **dúvidas comuns** e **FAQs**.
      - `agent_politicas_e_procedimentos`: Para informações sobre **garantia**, **prazos** e **regras internas**.
      - `agent_tickets`: Para consultas sobre o **status de tickets de atendimento**.

4.  **Recuperação e Geração de Resposta**: O agente especialista utiliza seu `retriever` para buscar o contexto relevante nas bases de dados (arquivos PDF e Excel) e, em seguida, usa um modelo de linguagem (LLM) para gerar uma resposta precisa e contextualizada.

5.  **Atualização do Histórico**: A resposta do agente é adicionada ao histórico da conversa, preparando o sistema para a próxima interação.

## 📁 Estrutura do Projeto

  - `app.py`: O frontend da aplicação, criado com Streamlit, que gerencia a interface do chat e a interação com o workflow.
  - `workflow.py`: Define o grafo de estados (`StateGraph`) e a lógica de roteamento do LangGraph, conectando o supervisor aos agentes especialistas.
  - `supervisor.py`: Contém a lógica do agente que atua como roteador, decidindo qual especialista deve ser acionado com base na pergunta do usuário.
  - `agents.py`: Define a classe `State` e as funções para cada agente especialista (`agent_detalhe_tecnico`, `agent_perguntas_e_respostas`, etc.). Cada função chama o agente principal (`agent_with_retriever`) com um prompt e um retriever específicos.
  - `retrievers.py`: Responsável por carregar e processar os dados dos documentos (PDFs e arquivos Excel), criando e gerenciando os `retrievers` vetoriais usando FAISS e HuggingFace Embeddings.
  - `config.py`: Gerencia a configuração do modelo de linguagem (LLM), utilizando a API do Gemini.
  - `docs/`: Pasta com os arquivos de contexto usados pelos retrievers (manuais, FAQs, políticas, e tickets).

## 🚀 Como Executar

Para rodar o projeto localmente, siga os passos abaixo:

1.  **Clone o repositório**:

    ```bash
    git clone <URL_DO_SEU_REPOSITÓRIO>
    cd <NOME_DO_SEU_PROJETO>
    ```

2.  **Instale as dependências**:
    Crie um ambiente virtual e instale as bibliotecas necessárias.

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure a chave de API**:
    Crie um arquivo `.env` na raiz do projeto e adicione sua chave de API do Google.

    ```bash
    GOOGLE_API_KEY="SUA_CHAVE_DE_API_AQUI"
    ```

4.  **Execute a aplicação Streamlit**:

    ```bash
    streamlit run app.py
    ```

O Atendente Virtual estará disponível em seu navegador local.