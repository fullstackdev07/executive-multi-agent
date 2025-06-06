def format_agent_output(
    title: str,
    sections: list,
    summary: str = "",
    next_steps: list = None,
    used_files: list = None,
    meta: dict = None
) -> dict:
    """
    Standardizes agent output for chat rendering and downstream use.
    - title: string, main title of the output
    - sections: list of dicts: [{"header": ..., "content": ...}]
    - summary: string, optional summary or context
    - next_steps: list of strings, suggested next actions
    - used_files: list of filenames
    - meta: dict, any extra info (agent name, timestamp, etc)
    """
    return {
        "title": title,
        "sections": sections,
        "summary": summary,
        "next_steps": next_steps or [],
        "used_files": used_files or [],
        "meta": meta or {},
    }

# Optional: helper to format sections as markdown

def format_sections_markdown(sections: list) -> str:
    """
    Converts sections list to a markdown string for display.
    """
    md = []
    for sec in sections:
        md.append(f"### {sec.get('header','')}")
        md.append(sec.get('content',''))
        md.append("")
    return "\n".join(md) 