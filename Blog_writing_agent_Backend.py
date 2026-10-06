import os
import re
import time
import operator
from pathlib import Path
from typing import TypedDict, Annotated, List, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from groq import RateLimitError, APIConnectionError, APITimeoutError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from PIL import Image, ImageDraw, ImageFont
from textwrap import wrap

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY_2")

# ---------------------------------------------------------------------------
# LLMs
# Planner/router need more output room (structured JSON); workers write sections.
# image_llm needs the most room: it returns the WHOLE blog again with placeholders.
# reasoning_effort="low" cuts hidden reasoning tokens, which count toward TPM.
# ---------------------------------------------------------------------------
planner_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=API_KEY,
    max_tokens=3000,
    max_retries=2,
    timeout=120,
    reasoning_effort="low",
)

worker_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=API_KEY,
    max_tokens=1800,
    max_retries=2,
    timeout=120,
    reasoning_effort="low",
)

image_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=API_KEY,
    max_tokens=6000,
    max_retries=2,
    timeout=180,
    reasoning_effort="low",
)

# Optional: if 8k TPM keeps throttling you, swap worker_llm for a lighter model:
# worker_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=API_KEY,
#                       max_tokens=1200, max_retries=2, timeout=120)


# def call_with_retry(fn, retries: int = 8):
#     """Run fn(); on Groq 429 wait out the per-minute window and retry."""
#     for attempt in range(retries):
#         try:
#             return fn()
#         except RateLimitError:
#             wait = min(10 * (attempt + 1), 60)
#             print(    f"[rate limit] waiting {wait}s (attempt {attempt + 1}/{retries})")
#             time.sleep(wait)
#     raise RuntimeError("Rate limit retries exhausted")

def call_with_retry(fn, retries: int = 8):
    """Retry temporary API rate-limit and connection failures."""
    for attempt in range(retries):
        try:
            return fn()

        except RateLimitError:
            wait = min(10 * (attempt + 1), 60)

            print(
                    f"[rate limit] waiting {wait}s "
                    f"(attempt {attempt + 1}/{retries})"
            )

            time.sleep(wait)

        except APITimeoutError:
            wait = min(5 * (attempt + 1), 30)

            print(
                    f"[timeout] waiting {wait}s "
                    f"(attempt {attempt + 1}/{retries})"
            )

            time.sleep(wait)

        except APIConnectionError:
            wait = min(5 * (attempt + 1), 30)

            print(
                    f"[connection] waiting {wait}s "
                    f"(attempt {attempt + 1}/{retries})"
            )

            time.sleep(wait)

    raise RuntimeError("API retries exhausted")


def safe_filename(title: str) -> str:
    """Filesystem- AND markdown-URL-safe slug: letters, digits, '_' and '-' only (no parentheses, commas, fancy hyphens)."""
    cleaned = re.sub(r"[\u2010-\u2015\u2212]", "-", title).strip().lower()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^\w\-]+", "", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned[:100] or "blog"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3-5 concrete non-overlapping subpoints to cover this section",
    )
    target_words: int = Field(..., description="Target word count for this section (120-450)")
    section_type: Literal[
        "intro", "core", "example", "checklist", "common_mistake", "conclusion"
    ] = Field(..., description="Use 'common_mistake' exactly once in plan")
    tags: List[str] = Field(default_factory=list)
    required_research: bool = False
    required_citation: bool = False
    required_code: bool = False


class Plan(BaseModel):
    blog_title: str
    tasks: List[Task]
    tone: str = Field(..., description="Writing tone (e.g. Practical, crisp)")
    audience: str = Field(..., description="Who this blog is for")
    blog_kind: Literal[
        "explainer", "system design", "tutorial", "news_roundup", "comparison"
    ] = "explainer"
    constraints: List[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class DiagramNode(BaseModel):
    id: str = Field(
        ...,
        description="Unique node id, e.g. router"
    )
    label: str = Field(
        ...,
        description="Exact readable text to display inside the diagram node"
    )


class DiagramEdge(BaseModel):
    source: str = Field(
        ...,
        description="Source node id"
    )
    target: str = Field(
        ...,
        description="Target node id"
    )
    label: Optional[str] = Field(
        default=None,
        description="Optional short label on the arrow"
    )


class DiagramSpec(BaseModel):
    title: str
    nodes: List[DiagramNode] = Field(default_factory=list)
    edges: List[DiagramEdge] = Field(default_factory=list)

class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model")

    image_type: Literal[
        "diagram",
        "flowchart",
        "architecture",
        "illustration"
    ] = Field(
        "illustration",
        description=(
            "Use diagram, flowchart, or architecture for technical "
            "structures. Use illustration only for conceptual/decorative visuals."
        )
    )
    diagram: Optional[DiagramSpec] = None

    size: Literal[
        "1024x1024",
        "1024x1536",
        "1536x1024",
        "1024x768",
    ] = "1024x1024"


    quality: Literal["low", "medium", "high"] = "medium"



class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)


class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    sections: Annotated[list[tuple[int, str]], operator.add]
    final: str

    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3-6 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
"""


def router_node(state: State) -> dict:
    decider = planner_llm.with_structured_output(RouterDecision)
    decision = call_with_retry(
        lambda: decider.invoke(
            [
                SystemMessage(content=ROUTER_SYSTEM),
                HumanMessage(content=   f"Topic: {state['topic']}"),
            ]
        )
    )
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }


def route_next(state: State):
    return "research" if state["needs_research"] else "orchestrator"


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------
def _tavily_search(query: str, max_results: int = 5):
    tool = TavilySearch(max_results=max_results)
    result = tool.invoke({"query": query})

    normalized = []
    for r in result.get("results", []):
        normalized.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": (r.get("content", "") or "")[:300],  # keep prompt small
                "published_at": r.get("published_date") or "",
                "source": r.get("source") or "",
            }
        )
    return normalized


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
- Return at most 12 items.
"""


def research_node(state: State):
    queries = state.get("queries", []) or []
    max_results = 4
    raw_results: list[dict] = []

    for q in queries:
        try:
            raw_results.extend(_tavily_search(q, max_results=max_results))
        except Exception as e:
            print(  f"[tavily] query failed: {q!r} -> {e}")

    if not raw_results:
        return {"evidence": []}

    extractor = planner_llm.with_structured_output(EvidencePack)
    pack = call_with_retry(
        lambda: extractor.invoke(
            [
                SystemMessage(content=RESEARCH_SYSTEM),
                HumanMessage(content=   f"Raw Results:\n{raw_results}"),
            ]
        )
    )

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    return {"evidence": list(dedup.values())}


# ---------------------------------------------------------------------------
# Orchestrator (planner)
# ---------------------------------------------------------------------------
ORCHESTRATOR_SYSTEM = (
    "You are a senior technical writer and developer advocate. "
    "Your job is to produce a highly actionable outline for a technical blog post.\n\n"
    "HARD REQUIREMENTS:\n"
    "- Create exactly 5-7 sections (tasks) that fit a technical blog.\n"
    "- Task ids must be unique integers starting at 1, in reading order.\n"
    "- EACH task MUST contain between 3 and 5 bullets.\n"
    "- NEVER generate 6 or more bullets for any task.\n"
    "- Before returning the final answer, COUNT the bullets in every task "
    "and make sure each task has 3, 4, or 5 bullets.\n"
    "- Each section must include:\n"
    "  1) goal - one sentence explaining what the reader should be able "
    "to do or understand after the section.\n"
    "  2) 3-5 concrete, specific, non-overlapping bullets.\n"
    "  3) target word count between 120 and 450.\n"
    "- Include EXACTLY ONE section with section_type='common_mistake'.\n\n"
    "MAKE IT TECHNICAL, NOT GENERIC:\n"
    "- Assume the reader is a developer; use correct terminology.\n"
    "- Prefer this engineering structure: "
    "problem -> intuition -> approach -> implementation -> "
    "trade-offs -> testing/observability -> conclusion.\n"
    "- Bullets must be actionable and testable.\n"
    "- Avoid vague bullets such as 'Explain X' or 'Discuss Y'.\n"
    "- Every bullet should clearly state what to build, compare, measure, "
    "implement, test, or verify.\n\n"
    "THE PLAN MUST INCLUDE AT LEAST ONE OF THESE:\n"
    "- A minimal working example (MWE) or code sketch.\n"
    "- Edge cases or failure modes.\n"
    "- Performance or cost considerations.\n"
    "- Security or privacy considerations when relevant.\n"
    "- Debugging or observability guidance such as logs, metrics, or traces.\n\n"
    "ORDERING GUIDANCE:\n"
    "- Start with a crisp introduction and problem framing.\n"
    "- Build core concepts before advanced details.\n"
    "- Include one common-mistakes section.\n"
    "- End with a practical checklist, summary, or next steps.\n\n"
    "FINAL VALIDATION BEFORE OUTPUT:\n"
    "- 5-7 total tasks.\n"
    "- Every task has 3-5 bullets.\n"
    "- No task has more than 5 bullets.\n"
    "- Exactly one task has section_type='common_mistake'.\n"
    "- Every task contains all required fields from the schema.\n\n"
    "Output must strictly match the Plan schema."
)


def orchestrator(state: State):
    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    planner = planner_llm.with_structured_output(Plan)
    plan = call_with_retry(
        lambda: planner.invoke(
            [
                SystemMessage(content=ORCHESTRATOR_SYSTEM),
                HumanMessage(
                    content=(
                            f"Topic: {state['topic']}\n"
                            f"Mode: {mode}\n\n"
                            f"Evidence (ONLY use for fresh claims; may be empty):\n"
                            f"{[e.model_dump() for e in evidence][:12]}"
                    )
                ),
            ]
        )
    )
    return {"plan": plan}


# ---------------------------------------------------------------------------
# Fan-out + worker
# ---------------------------------------------------------------------------
def fanout(state: State):
    evidence = [
        e.model_dump() if isinstance(e, EvidenceItem) else e
        for e in state.get("evidence", [])
    ]
    return [
        Send(
            "worker",
            {
                "task": task,
                "topic": state["topic"],
                "plan": state["plan"],
                "evidence": evidence,
                "mode": state.get("mode", "closed_book"),
            },
        )
        for task in state["plan"].tasks
    ]


WORKER_SYSTEM = (
    "You are an experienced engineer writing for other developers. "
    "Write ONE section of a technical blog post in Markdown.\n\n"

    "CORE RULES:\n"
    "- Follow the provided Goal.\n"
    "- Cover all Bullets in order.\n"
    "- Stay reasonably close to the requested target word count, but never add "
    "filler just to reach it.\n"
    "- Output ONLY the section in Markdown.\n"
    "- Start with exactly one '## <Section Title>' heading.\n"
    "- Do not add the blog's H1 title.\n\n"

    "HUMAN WRITING STYLE:\n"
    "- Write like a knowledgeable engineer explaining the idea to another "
    "engineer.\n"
    "- Prefer clear, direct sentences over polished marketing language.\n"
    "- Mix short and long sentences naturally.\n"
    "- Do not make every paragraph follow the same pattern.\n"
    "- Do not force the same number of paragraphs, bullets, or subsections "
    "into every section.\n"
    "- Use bullets only when they make the information easier to scan.\n"
    "- Use examples when they clarify an idea instead of adding generic "
    "examples everywhere.\n"
    "- Explain why a design choice matters when that context is useful.\n"
    "- Move quickly through obvious points and spend more detail on the "
    "parts that require reasoning.\n"
    "- Allow sections to have different structures depending on their purpose.\n\n"

    "AVOID AI-LIKE WRITING:\n"
    "- Do not open with phrases like 'In today's rapidly evolving world', "
    "'In the ever-changing landscape', or similar generic introductions.\n"
    "- Avoid phrases such as 'game-changing', 'cutting-edge', 'seamless', "
    "'robust solution', 'leverage', 'delve into', and 'revolutionary' unless "
    "the wording is genuinely necessary.\n"
    "- Do not repeat the same conclusion in multiple ways.\n"
    "- Do not add empty transition sentences just to connect paragraphs.\n"
    "- Do not create artificial summaries after every small section.\n"
    "- Do not turn simple explanations into unnecessarily formal prose.\n"
    "- Do not use exaggerated claims.\n"
    "- Do not invent statistics, benchmarks, personal experiences, or quotes.\n\n"

    "TECHNICAL QUALITY:\n"
    "- Be precise and implementation-oriented.\n"
    "- Prefer concrete APIs, data structures, protocols, and examples when "
    "they are relevant.\n"
    "- Explain trade-offs briefly when they actually matter.\n"
    "- Mention failure modes and edge cases when relevant.\n"
    "- Include a small code example only when code genuinely helps explain "
    "the section.\n"
    "- If code is included, keep it focused and internally consistent.\n\n"

    "GROUNDING:\n"
    "- Stay tightly focused on the requested topic.\n"
    "- Do not invent technologies, libraries, frameworks, databases, models, "
    "APIs, tools, architectures, benchmarks, or metrics.\n"
    "- Do not introduce a technology as part of the main solution unless it "
    "is directly relevant to the topic or supported by the provided evidence.\n"
    "- If an alternative technology is mentioned, label it clearly as an "
    "alternative or example.\n"
    "- Do not assume the reader's technology stack unless it is explicitly "
    "provided.\n"
    "- When research is supplied, use it for fresh factual claims and only "
    "cite URLs from the supplied evidence.\n"
    "- Never invent a source URL.\n"
    "- When a detail is uncertain, state the uncertainty instead of making "
    "up a specific fact.\n\n"

    "MARKDOWN STYLE:\n"
    "- Use normal paragraphs for explanations.\n"
    "- Use bullets for lists, steps, or comparisons where they genuinely help.\n"
    "- Use fenced code blocks for code.\n"
    "- Keep headings specific to the subject instead of generic headings "
    "such as 'Introduction' or 'Conclusion'.\n"
    "- Avoid unnecessary formatting.\n"
)


def worker(payload: dict) -> dict:
    task: Task = payload["task"]
    plan: Plan = payload["plan"]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    evidence = [
        e if isinstance(e, EvidenceItem) else EvidenceItem(**e)
        for e in payload.get("evidence", [])
    ]

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
                f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
            for e in evidence[:8]
        )

    messages = [
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(
            content=(
                    f"Blog: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog Kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section Title: {task.title}\n"
                    f"Section type: {task.section_type}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Bullets:{bullets_text}\n"
                    f"Tags: {task.tags}\n"
                    f"Required research: {task.required_research}\n"
                    f"Required citations: {task.required_citation}\n"
                    f"Required code: {task.required_code}\n"
                    f"Evidence (only use these URLs when citing):\n{evidence_text}\n"
            )
        ),
    ]

    section_md = call_with_retry(lambda: worker_llm.invoke(messages)).content.strip()
    print(  f"[worker] finished section {task.id}: {task.title}")

    return {"sections": [(task.id, section_md)]}


# ============================================================
# ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]

    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md =     f"# {plan.blog_title}\n\n{body}\n"

    # Always save the raw blog first so a failure in the image step never loses work.
    raw_path = Path(    f"{safe_filename(plan.blog_title)}_raw.md")
    raw_path.write_text(merged_md, encoding="utf-8")
    print(  f"[reducer] raw blog saved to {raw_path}")

    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding.
- Avoid decorative images when a technical diagram would be more useful.

IMAGE TYPE RULES:

1. "architecture"
   Use for system/component architecture.
   Example: User -> Router -> Retriever -> LLM.

2. "flowchart"
   Use for workflows, pipelines, processes, or state transitions.
   Example: START -> Router -> Research -> Planner -> Workers.

3. "diagram"
   Use for technical relationships, data flow, or conceptual structures.

4. "illustration"
   Use only for conceptual or decorative visuals where exact labels,
   boxes, arrows, or technical structure are NOT important.

IMPORTANT:
- Never use "illustration" for a technical architecture or workflow.
- Technical diagrams must contain exact, readable labels.
- For image_type diagram/flowchart/architecture you MUST fill the `diagram` field:
  nodes=[{id, label}] (3-8 nodes, labels max 4 words) and
  edges=[{source, target, label?}] using those node ids. Draw it top-to-bottom,
  no loops. If `diagram` is empty the image cannot be rendered.

- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan.
"""


def decide_images(state: State) -> dict:
    planner = image_llm.with_structured_output(GlobalImagePlan)
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    image_plan = call_with_retry(
        lambda: planner.invoke(
            [
                SystemMessage(content=DECIDE_IMAGES_SYSTEM),
                HumanMessage(
                    content=(
                            f"Blog kind: {plan.blog_kind}\n"
                            f"Topic: {state['topic']}\n\n"
                        "Insert placeholders + propose image prompts.\n\n"
                            f"{merged_md}"
                    )
                ),
            ]
        )
    )

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }

'''
REPLACEIGNG THE GEMINI MODEL WITH HUGGINGFACE MODEL FOR IMAGE GENERATION , BECAUSE GEMINI IS NOT PROVIDING THE FREE TIER IMAGE GENERATION..
'''


# def _gemini_generate_image_bytes(prompt: str) -> bytes:
#     """
#     Returns raw image bytes generated by Gemini.
#     Requires: pip install google-genai
#     Env var: GEMINI_API_KEY
#     """
#     from google import genai
#     from google.genai import types

#     api_key = os.environ.get("GEMINI_API_KEY")
#     if not api_key:
#         raise RuntimeError("GEMINI_API_KEY is not set.")

#     client = genai.Client(api_key=api_key)

#     resp = client.models.generate_content(
#         model="gemini-2.5-flash-image",
#         contents=prompt,
#         config=types.GenerateContentConfig(
#             response_modalities=["IMAGE"],
#             safety_settings=[
#                 types.SafetySetting(
#                     category="HARM_CATEGORY_DANGEROUS_CONTENT",
#                     threshold="BLOCK_ONLY_HIGH",
#                 )
#             ],
#         ),
#     )

#     # Depending on SDK version, parts may hang off resp.candidates[0].content.parts
#     parts = getattr(resp, "parts", None)
#     if not parts and getattr(resp, "candidates", None):
#         try:
#             parts = resp.candidates[0].content.parts
#         except Exception:
#             parts = None

#     if not parts:
#         raise RuntimeError("No image content returned (safety/quota/SDK change).")

#     for part in parts:
#         inline = getattr(part, "inline_data", None)
#         if inline and getattr(inline, "data", None):
#             return inline.data

#     raise RuntimeError("No inline image bytes found in response.")



def _hf_generate_image_bytes(prompt: str, size: str = "1024x1024") -> bytes:
    """
    Returns PNG bytes generated through Hugging Face Inference Providers.
    Requires: pip install huggingface_hub pillow
    Env vars: HUGGINGFACEHUB_ACCESS_TOKEN , (optional) HF_IMAGE_MODEL
    """
    import io
    from huggingface_hub import InferenceClient

    token = os.environ.get("HUGGINGFACEHUB_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACEHUB_ACCESS_TOKEN  is not set.")

    model = os.environ.get("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
    width, height = (int(x) for x in size.split("x"))

    client = InferenceClient(provider="auto", api_key=token)
    image = client.text_to_image(
        prompt,
        model=model,
        width=width,
        height=height,
    )  # returns a PIL.Image

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _render_diagram_png(diagram: DiagramSpec, out_path: Path):
    """
    Render a technical diagram from structured nodes + edges.

    This intentionally does NOT use an image model.
    Labels, nodes, and arrows are drawn programmatically so
    technical text stays exact and readable.
    """

    width = 1600
    node_width = 300
    node_height = 90
    vertical_gap = 150
    margin_y = 140

    nodes = diagram.nodes
    edges = diagram.edges

    if not nodes:
        raise ValueError("Diagram has no nodes.")

    # ---------------------------------------------------------
    # Simple automatic layout
    # ---------------------------------------------------------
    positions = {}

    # Build a simple dependency graph.
    incoming = {node.id: 0 for node in nodes}
    outgoing = {node.id: [] for node in nodes}

    for edge in edges:
        if edge.source in outgoing and edge.target in incoming:
            outgoing[edge.source].append(edge.target)
            incoming[edge.target] += 1

    # Find graph levels.
    levels = []
    remaining = [node.id for node in nodes]   # keep LLM order (set() order is random)

    while remaining:
        current_level = [
            node_id
            for node_id in remaining
            if incoming[node_id] == 0
        ]

        # Protect against malformed/cyclic diagrams.
        if not current_level:
            current_level = list(remaining)

        levels.append(current_level)

        for node_id in current_level:
            remaining.remove(node_id)

            for target in outgoing[node_id]:
                incoming[target] -= 1

    # Position nodes level by level.
    horizontal_gap = 100
    widest = max(len(lv) for lv in levels)
    width = max(width, widest * (node_width + horizontal_gap) + 200)

    for level_index, level_nodes in enumerate(levels):

        level_width = (
            len(level_nodes) * node_width
            + max(0, len(level_nodes) - 1) * horizontal_gap
        )

        start_x = (width - level_width) // 2

        y = margin_y + level_index * (
            node_height + vertical_gap
        )

        for node_index, node_id in enumerate(level_nodes):

            x = start_x + node_index * (
                node_width + horizontal_gap
            )

            positions[node_id] = (x, y)

    height = max(
        700,
        margin_y
        + len(levels) * (node_height + vertical_gap)
        + 100,
    )

    img = Image.new(
        "RGB",
        (width, height),
        "white",
    )

    draw = ImageDraw.Draw(img)

    # ---------------------------------------------------------
    # Fonts
    # ---------------------------------------------------------
    def _font(bold: bool, size: int):
        for path in (
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        ):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
        try:
            return ImageFont.load_default(size)   # Pillow >= 10.1: scalable default font
        except TypeError:
            return ImageFont.load_default()

    title_font = _font(True, 42)
    node_font = _font(True, 28)
    edge_font = _font(False, 22)

    # ---------------------------------------------------------
    # Title
    # ---------------------------------------------------------
    title = diagram.title.strip()

    title_bbox = draw.textbbox(
        (0, 0),
        title,
        font=title_font,
    )

    title_width = title_bbox[2] - title_bbox[0]

    draw.text(
        (
            (width - title_width) // 2,
            40,
        ),
        title,
        font=title_font,
        fill="black",
    )

    # ---------------------------------------------------------
    # Draw edges first
    # ---------------------------------------------------------

    for edge in edges:

        if edge.source not in positions:
            continue

        if edge.target not in positions:
            continue

        sx, sy = positions[edge.source]
        tx, ty = positions[edge.target]

        start_x = sx + node_width // 2
        start_y = sy + node_height

        end_x = tx + node_width // 2
        end_y = ty

        # Arrow line
        draw.line(
            (start_x, start_y, end_x, end_y),
            fill="black",
            width=5,
        )

        # -----------------------------------------------------
        # Arrow head
        # -----------------------------------------------------
        arrow_size = 14

        draw.polygon(
            [
                (end_x, end_y),
                (end_x - arrow_size, end_y - arrow_size * 2),
                (end_x + arrow_size, end_y - arrow_size * 2),
            ],
            fill="black",
        )

        # -----------------------------------------------------
        # Edge label
        # -----------------------------------------------------
        if edge.label:

            label = edge.label.strip()

            mid_x = (start_x + end_x) // 2
            mid_y = (start_y + end_y) // 2

            bbox = draw.textbbox(
                (0, 0),
                label,
                font=edge_font,
            )

            label_width = bbox[2] - bbox[0]
            label_height = bbox[3] - bbox[1]

            padding = 8

            draw.rounded_rectangle(
                (
                    mid_x - label_width // 2 - padding,
                    mid_y - label_height // 2 - padding,
                    mid_x + label_width // 2 + padding,
                    mid_y + label_height // 2 + padding,
                ),
                radius=8,
                fill="white",
                outline="black",
                width=2,
            )

            draw.text(
                (
                    mid_x - label_width // 2,
                    mid_y - label_height // 2,
                ),
                label,
                font=edge_font,
                fill="black",
            )

    # ---------------------------------------------------------
    # Draw nodes
    # ---------------------------------------------------------
    for node in nodes:

        x, y = positions[node.id]

        draw.rounded_rectangle(
            (
                x,
                y,
                x + node_width,
                y + node_height,
            ),
            radius=18,
            fill="white",
            outline="black",
            width=5,
        )

        # Wrap long labels
        lines = wrap(
            node.label.strip(),
            width=20,
        )

        line_heights = []

        for line in lines:
            bbox = draw.textbbox(
                (0, 0),
                line,
                font=node_font,
            )
            line_heights.append(
                bbox[3] - bbox[1]
            )

        total_height = sum(line_heights) + (
            len(lines) - 1
        ) * 8

        current_y = (
            y
            + node_height // 2
            - total_height // 2
        )

        for line, line_height in zip(
            lines,
            line_heights,
        ):
            bbox = draw.textbbox(
                (0, 0),
                line,
                font=node_font,
            )

            text_width = bbox[2] - bbox[0]

            draw.text(
                (
                    x
                    + node_width // 2
                    - text_width // 2,
                    current_y,
                ),
                line,
                font=node_font,
                fill="black",
            )

            current_y += line_height + 8

    img.save(
        out_path,
        format="PNG",
    )

    print(  f"[diagram] saved {out_path}")


# ---------------------------------------------------------------------------
# Image placement
# ---------------------------------------------------------------------------
GENERATE_IMAGES = (
    os.getenv(
        "GENERATE_IMAGES",
        "1",
    )
    == "1"
)


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]

    if plan is None:
        raise ValueError(
            "generate_and_place_images called without plan."
        )

    markdown = (
        state.get("md_with_placeholders")
        or state["merged_md"]
    )

    image_specs = (
        state.get("image_specs")
        or []
    )

    out_md_name = (
            f"{safe_filename(plan.blog_title)}.md"
    )

    if not image_specs:
        Path(out_md_name).write_text(
            markdown,
            encoding="utf-8",
        )
        return {
            "final": markdown
        }
    blog_folder = safe_filename(plan.blog_title)

    images_dir = Path("images") / blog_folder
    images_dir.mkdir(
    parents=True,
    exist_ok=True,
    )

    quota_hit = False

    for raw_spec in image_specs:

        spec = (
            raw_spec.model_dump()
            if isinstance(
                raw_spec,
                ImageSpec,
            )
            else raw_spec
        )

        placeholder = spec.get(
            "placeholder",
            "",
        )

        filename = (
            safe_filename(
                Path(
                    spec.get(
                        "filename",
                        "image.png",
                    )
                ).stem
            )
            + ".png"
        )

        out_path = (
            images_dir / filename
        )

        image_type = spec.get(
            "image_type",
            "illustration",
        )

        # -----------------------------------------------------
        # Technical diagram
        # -----------------------------------------------------
        if image_type in {
            "diagram",
            "flowchart",
            "architecture",
        }:

            diagram_data = spec.get(
                "diagram"
            )

            if not diagram_data:
                markdown = markdown.replace(
                    placeholder,
                    (
                            f"<!-- TODO diagram {filename}: "
                        "diagram specification missing -->"
                    ),
                )
                continue

            if not out_path.exists():

                try:
                    diagram = (
                        diagram_data
                        if isinstance(
                            diagram_data,
                            DiagramSpec,
                        )
                        else DiagramSpec(
                            **diagram_data
                        )
                    )

                    _render_diagram_png(
                        diagram,
                        out_path,
                    )

                except Exception as exc:
                    print(
                            f"[diagram] {filename} failed: {exc}"
                    )

                    markdown = markdown.replace(
                        placeholder,
                        (
                                f"<!-- TODO diagram {filename}: "
                                f"{exc} -->"
                        ),
                    )
                    continue

        # -----------------------------------------------------
        # Illustration -> Hugging Face
        # -----------------------------------------------------
        else:

            if not out_path.exists():

                if (
                    not GENERATE_IMAGES
                    or quota_hit
                ):
                    markdown = markdown.replace(
                        placeholder,
                        (
                                f"<!-- TODO image {filename}: "
                                f"{spec.get('prompt', '')} -->"
                        ),
                    )
                    continue

                try:
                    image_bytes = (
                        _hf_generate_image_bytes(
                            spec.get(
                                "prompt",
                                "",
                            ),
                            spec.get(
                                "size",
                                "1024x1024",
                            ),
                        )
                    )

                    out_path.write_bytes(
                        image_bytes
                    )

                    print(
                            f"[image] saved {out_path}"
                    )

                except Exception as exc:

                    error_text = str(exc)

                    short_error = (
                        error_text
                        .splitlines()[0][:150]
                        if error_text
                        else type(exc).__name__
                    )

                    print(
                            f"[image] {filename} failed: "
                            f"{short_error}"
                    )

                    lowered = (
                        error_text.lower()
                    )

                    if any(
                        key in lowered
                        for key in (
                            "402",
                            "429",
                            "credit",
                            "quota",
                            "rate limit",
                        )
                    ):
                        quota_hit = True

                    markdown = markdown.replace(
                        placeholder,
                        (
                                f"<!-- TODO image {filename}: "
                                f"{spec.get('prompt', '')} -->"
                        ),
                    )
                    continue

        # -----------------------------------------------------
        # Add local markdown image
        # -----------------------------------------------------
        if out_path.exists():

            image_markdown = (
        f"![{spec.get('alt', '')}]"
        f"(images/{blog_folder}/{filename})\n"
        f"*{spec.get('caption', '')}*"
        )

            markdown = markdown.replace(
                placeholder,
                image_markdown,
            )

    Path(out_md_name).write_text(
        markdown,
        encoding="utf-8",
    )

    return {
        "final": markdown
    }
    
# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
graph = StateGraph(State)
graph.add_node("router", router_node)
graph.add_node("research", research_node)
graph.add_node("orchestrator", orchestrator)
graph.add_node("worker", worker)
graph.add_node("reducer", reducer_subgraph)

graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router", route_next, {"research": "research", "orchestrator": "orchestrator"}
)
graph.add_edge("research", "orchestrator")
graph.add_conditional_edges("orchestrator", fanout, ["worker"])
graph.add_edge("worker", "reducer")
graph.add_edge("reducer", END)

app = graph.compile()


if __name__ == "__main__":
    result = app.invoke(
        {"topic": "Write a blog on Self Attention in transformer Architecture", "sections": []},
        config={"max_concurrency": 2},  # keeps parallel workers under the 8k TPM limit
    )
    print("\nDone. Blog length (chars):", len(result["final"]))
