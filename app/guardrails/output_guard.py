from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def verify_hallucination(context: str, answer: str) -> bool:
    """
    Uses an LLM call to verify if the answer is supported by the context.
    Returns True if Safe, False if Hallucinated.
    """
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    prompt = PromptTemplate(
        template="""
        You are a medical compliance officer. 
        Context: {context}
        Answer: {answer}
        
        Does the answer strictly rely on the provided context? 
        If the answer mentions facts not in the context, reply 'NO'.
        If it is safe and supported, reply 'YES'.
        Only reply YES or NO.
        """,
        input_variables=["context", "answer"]
    )
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"context": context, "answer": answer})
    
    return "YES" in result.strip().upper()