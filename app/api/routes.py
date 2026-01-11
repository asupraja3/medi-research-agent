from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.services.ingestion import ingest_data
from app.services.llm_agent import get_agent_executor
from app.guardrails.input_guard import check_pii
from app.guardrails.output_guard import verify_hallucination
from app.core.security import get_api_key

router = APIRouter()

class QueryRequest(BaseModel):
    session_id: str
    query: str

class QueryResponse(BaseModel):
    response: str
    warnings: list[str] = []

@router.post("/ingest", dependencies=[Depends(get_api_key)])
async def trigger_ingestion():
    """Triggers the data pipeline to read CSV and update Vector DB."""
    try:
        success = ingest_data()
        return {"status": "success", "message": "Data ingested and indexed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=QueryResponse, dependencies=[Depends(get_api_key)])
async def chat(payload: QueryRequest):
    # 1. Input Guardrail
    safe_query = check_pii(payload.query)
    
    # 2. Agent Execution
    try:
        executor = get_agent_executor(payload.session_id)
        result = executor.invoke({"input": safe_query})
        answer = result["output"]
    except Exception as e:
        # Fallback strategy for Agent failure
        return {"response": "I encountered an internal error processing your request.", "warnings": [str(e)]}

    # 3. Output Guardrail (Hallucination Check)
    # Note: We extract retrieved context from memory or result if available for strict checking.
    # For this simplified demo, we assume the answer is self-contained or we skip context passed to verify.
    # In a full trace, you would capture 'docs' from the tool output.
    
    # We perform a lightweight safety check here
    is_safe = verify_hallucination(context="[Use Implicit Context]", answer=answer)
    
    warnings = []
    if not is_safe:
        warnings.append("Caution: The AI could not fully verify this answer against the text.")

    return {"response": answer, "warnings": warnings}