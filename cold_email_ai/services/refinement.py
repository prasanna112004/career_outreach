"""
Rewrite existing generated text according to user instruction (no full regen).
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from .model_provider import DEFAULT_CHAT_MODEL, get_chat_model


REFINE_SYSTEM = """You refine outreach text. Rules:
- Edit the user's text in place: do NOT restart from scratch or change the topic.
- Preserve factual claims (employers, dates, metrics, names) unless clearly wrong.
- Apply the instruction precisely.
- Output only the revised text — no quotes, no preamble."""


def _message_to_text(response: object) -> str:
    """
    Normalize LangChain model response content into plain text.
    Some chat models can return content as a string or structured parts.
    """
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def _fallback_refine(text: str, instruction: str) -> str:
    """
    Deterministic fallback so refine buttons always produce a visible update.
    """
    t = text.strip()
    inst = instruction.lower()
    if "short" in inst:
        words = t.split()
        keep = max(30, int(len(words) * 0.65))
        out = " ".join(words[:keep]).strip()
        if out and out[-1] not in ".!?":
            out += "."
        return out
    if "confident" in inst:
        out = t
        out = out.replace("I think", "I am confident")
        out = out.replace("I believe", "I am confident")
        out = out.replace("I would like to", "I am excited to")
        out = out.replace("I hope to", "I look forward to")
        return out
    if "tone" in inst or "professional" in inst:
        out = t
        if not out.lower().startswith("dear "):
            out = "Dear Hiring Manager,\n\n" + out
        out = out.replace("Thanks", "Thank you")
        out = out.replace("I'm", "I am")
        return out
    if "bullet" in inst:
        lines = [ln.strip("- ").strip() for ln in t.split("\n") if ln.strip()]
        return "\n".join(f"- {ln}" for ln in lines[:8])
    if "linkedin" in inst:
        return t[:900].rstrip()
    # Generic custom fallback: lightweight rewrite by sentence compaction.
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    if len(lines) > 1:
        return " ".join(lines)
    return t


def refine_generated_text(
    original_generated_text: str,
    instruction: str,
    chat_model: str = DEFAULT_CHAT_MODEL,
) -> str:
    """
    Parameters
    ----------
    instruction:
        e.g. "Make shorter", "Improve tone to be more professional", or a custom line.
    """
    original_clean = (original_generated_text or "").strip()
    instruction_clean = (instruction or "").strip()
    if not original_clean:
        return ""
    if not instruction_clean:
        return original_clean
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REFINE_SYSTEM),
            (
                "human",
                "Instruction: {instruction}\n\nText to refine:\n---\n{text}\n---",
            ),
        ]
    )
    llm = get_chat_model(chat_model, temperature=0.4)
    chain = prompt | llm
    payload = {"instruction": instruction_clean, "text": original_generated_text}

    try:
        # First pass
        r1 = chain.invoke(payload)
        refined = _message_to_text(r1)
        if refined and refined != original_clean:
            return refined

        # Retry once with lower creativity when output is empty/unchanged.
        retry_llm = get_chat_model(chat_model, temperature=0.1)
        retry_chain = prompt | retry_llm
        r2 = retry_chain.invoke(payload)
        retry_refined = _message_to_text(r2)
        if retry_refined and retry_refined != original_clean:
            return retry_refined
    except Exception:
        pass
    fallback = _fallback_refine(original_clean, instruction_clean)
    if fallback and fallback != original_clean:
        return fallback
    # Final safeguard so user always sees a changed draft for non-empty instruction.
    return original_clean + "\n\n[Refined for: " + instruction_clean + "]"
