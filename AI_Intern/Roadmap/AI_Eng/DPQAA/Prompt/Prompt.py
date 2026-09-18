"""Contains prompts and instructions for the LLMs"""

from pathlib import Path

PROMPT_DIR = Path(__file__).parent

def load_prompt(path: str) -> str:
    """Load a prompt from a Markdown file."""
    return (PROMPT_DIR / path).read_text(encoding="utf-8").strip()

rules = load_prompt("base/common.md")
markdown_rules = load_prompt("base/markdown.md")
cite_rules = load_prompt("base/citation.md")
math_rules = load_prompt("base/math.md")
merge_rules = load_prompt("synthesis/merge.md")

history_resolver_prompt = load_prompt("routing/resolver.md")
query_parallelizer_prompt = load_prompt("routing/parallelizer.md")
fpt_summarizer_prompt = load_prompt("synthesis/fpt_summary.md")

chitchat_instruction = load_prompt("instructions/chitchat.md")
code_instruction = load_prompt("instructions/code.md")
code_solver_instruction = load_prompt("instructions/code_solver.md")
fpt_related_instruction = load_prompt("instructions/fpt_related.md")
fpt_internal_not_found_instruction = load_prompt("instructions/fpt_internal_not_found.md")
greeting_instruction = load_prompt("instructions/greeting.md")
other_instruction = load_prompt("instructions/outlaw.md")