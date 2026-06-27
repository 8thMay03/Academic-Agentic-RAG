def clean_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())

