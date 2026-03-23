"""Layer 3: Page context configurations.

No fake personas. Just SPARC being SPARC, aware of which page the user is on.
Each page config tells SPARC what domain to lead with and which signals to prioritize.

Built from: The SPARC Site Architecture (Final Rebrand).docx
"""

from dataclasses import dataclass


@dataclass
class PageContext:
    key: str
    page_name: str
    route: str
    opening_line: str
    lead_with: str
    signal_focus: list[str]
    domain_topics: list[str]


PAGES: dict[str, PageContext] = {
    "analyst": PageContext(
        key="analyst",
        page_name="Strategy & Intelligence",
        route="/services/strategy-intelligence",
        opening_line="Strategy without intelligence is noise. What signals are you reading?",
        lead_with="strategic analysis, competitive positioning, and demand signal mapping",
        signal_focus=["visibility", "momentum", "authority"],
        domain_topics=[
            "Signal Intelligence™ and pattern recognition",
            "Competitor analysis and market positioning",
            "Demand signal mapping and search behavior",
            "Strategic roadmapping",
            "Data interpretation and Early-Shift Detection™",
        ],
    ),

    "navigator": PageContext(
        key="navigator",
        page_name="Visibility & Authority",
        route="/services/visibility-authority",
        opening_line="Visibility gets attention. Authority earns trust. Tell me where you stand.",
        lead_with="search visibility, authority building, and AI-driven discovery",
        signal_focus=["visibility", "authority", "trust"],
        domain_topics=[
            "SEO (Technical, Local, E-commerce)",
            "GEO (Generative Engine Optimization)",
            "AEO (Answer Engine Optimization)",
            "Entity SEO and Schema",
            "Content authority and backlinks",
            "Digital PR",
            "Google Business Profile",
        ],
    ),

    "accelerator": PageContext(
        key="accelerator",
        page_name="Acquisition & Conversion",
        route="/services/acquisition-conversion",
        opening_line="Growth should be predictable, not chaotic. Where is your conversion leaking?",
        lead_with="paid acquisition, conversion systems, and revenue efficiency",
        signal_focus=["conversion", "momentum", "visibility"],
        domain_topics=[
            "Paid Media (Google Ads, Meta Ads, Microsoft Ads)",
            "Retargeting and remarketing",
            "Landing page and funnel optimization",
            "Conversion Rate Optimization (CRO)",
            "CAC, ROAS, and lead generation systems",
        ],
    ),

    "engineer": PageContext(
        key="engineer",
        page_name="Experience & Infrastructure",
        route="/services/experience-infrastructure",
        opening_line="Trust erodes when systems fail. How stable is your infrastructure?",
        lead_with="technical infrastructure, platform health, and system reliability",
        signal_focus=["trust", "conversion", "engagement"],
        domain_topics=[
            "Web Design and Development",
            "Hosting, speed, Core Web Vitals",
            "UI/UX design and mobile experience",
            "CRM and automation systems",
            "Analytics infrastructure (GA4, GTM, Looker Studio)",
            "Technical health and security",
        ],
    ),

    "multiplier": PageContext(
        key="multiplier",
        page_name="Retention & Growth",
        route="/services/retention-growth",
        opening_line="The sale is not the finish line. It is the starting signal. What happens after?",
        lead_with="retention, lifecycle value, and compounding growth systems",
        signal_focus=["momentum", "trust", "engagement"],
        domain_topics=[
            "Email marketing and automation",
            "SMS and direct messaging",
            "CRM and lifecycle management",
            "Customer retention and CLV",
            "Review generation and reputation",
            "Referral and loyalty systems",
        ],
    ),

    "connector": PageContext(
        key="connector",
        page_name="About / Home",
        route="/about",
        opening_line="SPARC active. Every click, search, and mention is a signal. How can I help?",
        lead_with="the Power Spark X™ ecosystem and how all signals interconnect",
        signal_focus=["visibility", "authority", "trust", "engagement", "conversion", "momentum"],
        domain_topics=[
            "The Power Spark X™ ecosystem",
            "Lead Clickz as the strategic authority",
            "The 6 Spark Signals™ and their relationships",
            "The 5 service pillars",
            "How SPARC™ diagnoses and directs strategy",
        ],
    ),
}


def get_persona(persona_key: str) -> PageContext:
    """Get a page context by key, falling back to connector."""
    return PAGES.get(persona_key, PAGES["connector"])
