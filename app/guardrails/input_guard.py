import re

def check_pii(text: str) -> str:
    """
    Simple Regex Guardrail to detect and redact SSNs or Phone Numbers.
    """
    # Pattern for US Phone numbers
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    # Pattern for SSN
    ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
    
    text = re.sub(phone_pattern, "[PHONE REDACTED]", text)
    text = re.sub(ssn_pattern, "[SSN REDACTED]", text)
    
    return text