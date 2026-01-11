from langchain.tools import tool

@tool
def check_drug_interactions(drug_a: str, drug_b: str) -> str:
    """
    Checks for adverse interactions between two drugs.
    Use this tool whenever a user asks about medication safety.
    """
    # Mock Database of interactions
    # In production, this would call openFDA API
    interactions = {
        ("aspirin", "warfarin"): "SEVERE: Increased risk of bleeding.",
        ("ibuprofen", "lisinopril"): "MODERATE: May reduce antihypertensive effect.",
    }
    
    pair = tuple(sorted((drug_a.lower(), drug_b.lower())))
    
    if pair in interactions:
        return f"Interaction Found: {interactions[pair]}"
    else:
        return "No major interactions found in the database for these two drugs."