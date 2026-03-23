"""Prompt builder — assembles the 3-layer system prompt.

Layer 1: SPARC Core Identity (constant)
Layer 2: Knowledge Base Context (from RAG retrieval)
Layer 3: Page Context (which page the user is on — affects domain focus, not personality)
"""

from backend.prompts.core_identity import SPARC_CORE_IDENTITY
from backend.prompts.personas import PageContext, get_persona
from backend.config import DEFAULT_PERSONA, ROUTE_TO_PERSONA


def resolve_persona_key(page_route: str | None) -> str:
    """Map a page route to a page context key."""
    if not page_route:
        return DEFAULT_PERSONA
    # Normalize: strip trailing slash, lowercase
    route = page_route.rstrip("/").lower()
    return ROUTE_TO_PERSONA.get(route, DEFAULT_PERSONA)


def build_system_prompt(
    persona_key: str,
    rag_context: str = "",
) -> str:
    """Assemble the full 3-layer system prompt.

    Args:
        persona_key: Which page context to use (analyst, navigator, etc.)
        rag_context: Retrieved knowledge base chunks formatted as text.

    Returns:
        Complete system prompt string.
    """
    page = get_persona(persona_key)

    # Layer 1: Core Identity
    prompt_parts = [SPARC_CORE_IDENTITY]

    # Layer 2: Knowledge Base Context (RAG)
    if rag_context:
        prompt_parts.append(_format_rag_layer(rag_context))

    # Layer 3: Page Context
    prompt_parts.append(_format_page_layer(page))

    return "\n\n".join(prompt_parts)


def _format_rag_layer(rag_context: str) -> str:
    """Format the RAG context as Layer 2."""
    return f"""═══════════════════════════════════════
RELEVANT KNOWLEDGE BASE CONTEXT
═══════════════════════════════════════

The following information has been retrieved from the SPARC™ knowledge base to help answer the current query. Use this context to inform your response, but integrate it naturally — do not quote it verbatim or reference it as "retrieved context."

{rag_context}"""


def _format_page_layer(page: PageContext) -> str:
    """Format the page context as Layer 3."""
    domain_items = ", ".join(page.domain_topics)
    signal_items = ", ".join(page.signal_focus)

    return f"""═══════════════════════════════════════
CURRENT PAGE: {page.page_name}
═══════════════════════════════════════

The user is currently on the {page.page_name} page.

When this conversation starts, your opening line is:
"{page.opening_line}"

DOMAIN FOCUS FOR THIS PAGE:
The user came to this page to learn about {page.lead_with}. Lead your diagnosis from this angle. Your primary domain topics here are: {domain_items}.

SIGNAL PRIORITY: {signal_items}
When diagnosing, check these signals first — but always connect them to the full ecosystem.

You still have access to ALL knowledge across Power Spark X™. If the user asks about something outside this page's domain, answer it fully using the same SPARC voice. Do not redirect them or say "that's covered on another page." Just diagnose it."""
