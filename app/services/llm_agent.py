from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from app.services.retrieval import get_hybrid_retriever
from app.tools.drug_checker import check_drug_interactions

# Global store for session memories
# In production, use Redis or a Database
session_store = {}

def get_agent_executor(session_id: str):
    # 1. Setup Tools
    retriever = get_hybrid_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "search_medical_history",
        "Searches past patient cases and medical transcriptions. Use this for finding similar symptoms or treatments."
    )
    
    tools = [retriever_tool, check_drug_interactions]

    # 2. Setup LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 3. Setup Prompt (System Prompt Engineering)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful Medical Assistant. "
                   "You must strictly use your tools to answer questions. "
                   "If you use the search tool, base your answer ONLY on the retrieved results. "
                   "Do not make up medical advice."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        MessagesPlaceholder(variable_name="chat_history"),
    ])

    # 4. Setup Memory (Context Management)
    if session_id not in session_store:
        session_store[session_id] = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            max_token_limit=2000 # Fallback strategy for long context
        )
    memory = session_store[session_id]

    # 5. Create Agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        memory=memory, 
        verbose=True,
        handle_parsing_errors=True # Fallback logic
    )
    
    return agent_executor