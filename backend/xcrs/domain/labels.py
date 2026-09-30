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
}

# Easier to maintain as text than as a 90-line list.
WORDS_TEXT = """
AI API APIs AWS CD CDN CI CLI CORS CSS CSV DNS DOM FTP GCP GPU GraphQL GitHub GitLab HTML HTTP
HTTPS IDE iOS IP JavaScript JSON JWT LLM LLMs ML MongoDB MySQL NLP NoSQL npm OAuth ORM OS PHP
PostgreSQL QA REST SDK SOLID SQL SSH SSL TCP TDD TLS TypeScript UDP UI URL UX VCS XML YAML
WebSockets OpenGL DevOps SQLite Redis Linux Kubernetes Docker Git Python Java Kotlin Rust Go Ruby
Swift Unity Unreal Android React Angular Vue Svelte Figma Jenkins Terraform Ansible Azure Selenium
Cypress Jest Solidity Ethereum EC2 S3 ECS EKS MSSQL CLT IaC gRPC RabbitMQ Kafka Nginx
"""
WORDS = {w.lower(): w for w in WORDS_TEXT.split()}


def display_label(name: str) -> str:
    """ "ci cd" → "CI/CD", "nosql databases" → "NoSQL databases", "event sourcing" → "Event sourcing"."""
    text = " ".join(name.split())
    if text.lower() in PHRASES:
        return PHRASES[text.lower()]
    parts = re.split(r"(\s+)", text)
    known_first = parts[0].lower() in WORDS  # keep "npm", "iOS" as they are
    label = "".join(WORDS.get(w.lower(), w) for w in parts)
    return label if known_first or not label else label[:1].upper() + label[1:]
