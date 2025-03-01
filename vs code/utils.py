import huggingface_hub

def validate_token(token: str) -> bool:
    """Validate Hugging Face token."""
    try:
        huggingface_hub.HfApi().whoami(token=token)
        return True
    except Exception:
        return False