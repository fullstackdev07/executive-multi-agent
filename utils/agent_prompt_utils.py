UNIVERSAL_AGENT_GUIDANCE = """
If the input is unclear, incomplete, or missing essential sections (such as job spec, CV, or consultant assessment), respond with:

“I’m sorry, I couldn’t understand or generate a report from the provided information. Please ensure all required sections are included and clearly labeled.”

Do not assume or hallucinate missing content. Only use the information explicitly provided.
"""

def inject_guidance(template_body: str) -> str:
    """
    Utility to prepend universal guidance to a prompt template body.
    """
    return f"{UNIVERSAL_AGENT_GUIDANCE}\n\n{template_body}"