"""Readable labels for roadmap node names. Pure functions.

roadmap.sh node names are lowercase file slugs ("ci cd", "csharp", "nosql databases"). Suggestions show
them to people, so known technical words get their usual spelling. Roadmaps we generate ourselves
should carry proper display labels instead (see the data-quality notes).
"""

import re

PHRASES = {
    "ci cd": "CI/CD",
    "csharp": "C#",
    "cpp": "C++",
    "c plus plus": "C++",
    "dot net": ".NET",
    "node js": "Node.js",
    "nodejs": "Node.js",
    "vue js": "Vue.js",
    "next js": "Next.js",
    "ab testing": "A/B testing",
    "gke eks aks": "GKE / EKS / AKS",
}

# "<name> js" slugs. Libraries named in lowercase (ethers.js, web3.js) keep it.
JS_NAMES = {
    "next": "Next.js",
    "nuxt": "Nuxt.js",
    "vue": "Vue.js",
    "node": "Node.js",
    "express": "Express.js",
    "solid": "SolidJS",
    "after": "After.js",
    "zombie": "Zombie.js",
}

# Easier to maintain as text than as a 90-line list.
WORDS_TEXT = """
AI API APIs AWS CD CDN CI CLI CORS CSS CSV DNS DOM FTP GCP GPU GraphQL GitHub GitLab HTML HTTP
HTTPS IDE iOS IP JavaScript JSON JWT LLM LLMs ML MongoDB MySQL NLP NoSQL npm OAuth ORM OS PHP
PostgreSQL QA REST SDK SOLID SQL SSH SSL TCP TDD TLS TypeScript UDP UI URL UX VCS XML YAML
WebSockets OpenGL DevOps SQLite Redis Linux Kubernetes Docker Git Python Java Kotlin Rust Go Ruby
Swift Unity Unreal Android React Angular Vue Svelte Figma Jenkins Terraform Ansible Azure Selenium
Cypress Jest Solidity Ethereum EC2 S3 ECS EKS MSSQL CLT IaC gRPC RabbitMQ Kafka Nginx ORMs GKE AKS
"""
WORDS = {w.lower(): w for w in WORDS_TEXT.split()}


def display_label(name: str) -> str:
    """ "ci cd" → "CI/CD", "nosql databases" → "NoSQL databases", "event sourcing" → "Event sourcing"."""
    text = " ".join(name.split())
    if text.lower() in PHRASES:
        return PHRASES[text.lower()]
    if js := re.fullmatch(r"(\w+) js", text.lower()):
        return JS_NAMES.get(js[1], f"{js[1]}.js")
    parts = re.split(r"(\s+)", text)
    known_first = parts[0].lower() in WORDS  # keep "npm", "iOS" as they are
    label = "".join(WORDS.get(w.lower(), w) for w in parts)
    return label if known_first or not label else label[:1].upper() + label[1:]


# Roadmap section names that aren't skills, and phrasings that wrap one ("learn dom manipulation").
NOT_A_SKILL = re.compile(
    r"^(learn the basics|the basics|basics|introduction|getting started|how does .+ work"
    r"|checkpoint\b.*|more about\b.*)$"
)
SKILL_PREFIX = re.compile(
    r"^(learn the |learn |basics of |basic usage of |what is an? |what is |what are |understanding |understand )"
)


def skill_label(name: str) -> str | None:
    """The label to suggest as a skill: prefixes removed ("what is http" → "HTTP"), filler dropped (None)."""
    text = " ".join(name.lower().split())
    if NOT_A_SKILL.match(text):
        return None
    text = SKILL_PREFIX.sub("", text, count=1)
    return display_label(text) if text else None
