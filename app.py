import importlib
import io
import os
import re
import sys
import traceback
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BACKEND_MODULE = os.getenv("BLOG_BACKEND_MODULE", "Blog_writing_agent_Enhanced")
_HERE = Path(__file__).resolve().parent
# app.py may sit next to the backend OR in a sub-folder such as <project>/frontend/app.py.
if os.getenv("BLOG_PROJECT_ROOT"):
    PROJECT_ROOT = Path(os.environ["BLOG_PROJECT_ROOT"]).resolve()
elif (_HERE / f"{BACKEND_MODULE}.py").exists():
    PROJECT_ROOT = _HERE
else:
    PROJECT_ROOT = _HERE.parent
MAX_BLOGS = 50
MAX_CONCURRENCY = 2  # matches your backend's rate-limit-safe setting

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)


# -----------------------------------------------------------------------------
# Backend loader
# -----------------------------------------------------------------------------
def load_backend():
    """Import the compiled LangGraph app. Re-executes the module only if its source file changed."""
    module = importlib.import_module(BACKEND_MODULE)
    path = getattr(module, "__file__", None)
    mtime = os.path.getmtime(path) if path else None
    if getattr(module, "_loaded_mtime", mtime) != mtime:
        module = importlib.reload(module)
    module._loaded_mtime = mtime
    if not hasattr(module, "app"):
        raise RuntimeError(
            f"Backend module '{BACKEND_MODULE}' does not expose a compiled 'app'."
        )
    return module


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def model_to_dict(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return [model_to_dict(x) for x in value]
    return value


def safe_slug(title: str) -> str:
    """Same rule the backend uses for <slug>.md / <slug>_raw.md / images/<slug>/ (so lookups match)."""
    try:
        return load_backend().safe_filename(title)
    except Exception:
        s = re.sub(r"[\u2010-\u2015\u2212]", "-", title).strip().lower()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^\w\-]+", "", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s[:100] or "blog"


def extract_title(md: str, fallback: str = "blog") -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


def list_past_blogs() -> list[Path]:
    files = [
        p for p in PROJECT_ROOT.glob("*.md")
        if p.is_file() and not p.name.endswith("_raw.md")
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:MAX_BLOGS]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def count_workers(state: dict[str, Any]) -> tuple[int, int]:
    sections = state.get("sections") or []
    ids = set()
    for item in sections:
        try:
            task_id = int(item[0])
            ids.add(task_id)
        except (TypeError, ValueError, IndexError):
            continue

    plan = model_to_dict(state.get("plan")) or {}
    total = len(plan.get("tasks") or [])
    return min(len(ids), total), total




STAGE_LABELS = {
    "router": "Router",
    "research": "Research",
    "orchestrator": "Planner",
    "worker": "Writers",
    "merge_content": "Merge",
    "decide_images": "Image Planning",
    "generate_and_place_images": "Image Generation",
}

def running_stage(last_node: str | None, state: dict[str, Any]) -> str:
    """Show the next stage after the node that actually finished."""
    if not last_node:
        return "router"
    if last_node == "router":
        return "research" if state.get("needs_research") else "orchestrator"
    if last_node == "research":
        return "orchestrator"
    if last_node == "orchestrator":
        return "worker"
    if last_node == "worker":
        done, total = count_workers(state)
        return "worker" if total and done < total else "merge_content"
    if last_node == "merge_content":
        return "decide_images"
    if last_node == "decide_images":
        return "generate_and_place_images"
    if last_node == "generate_and_place_images":
        return "done"
    return last_node


def stage_label(stage: str) -> str:
    return STAGE_LABELS.get(stage, stage.replace("_", " ").title())


def referenced_images(md: str) -> list[Path]:
    """Return only the image files referenced by the current blog markdown."""
    refs = re.findall(r"!\[[^\]]*\]\(((?:[^()\s]|\([^()]*\))+)\)", md)
    files: list[Path] = []
    seen: set[Path] = set()

    for ref in refs:
        if ref.startswith(("http://", "https://")):
            continue
        rel = ref.strip().lstrip("./")
        path = (PROJECT_ROOT / rel).resolve()
        try:
            valid = path.is_file() and path.is_relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            valid = False
        if valid and path not in seen:
            files.append(path)
            seen.add(path)
    return files


def bundle_zip(md: str, md_filename: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(md_filename, md.encode("utf-8"))
        for image_path in referenced_images(md):
            zf.write(image_path, arcname=image_path.relative_to(PROJECT_ROOT).as_posix())
    return buf.getvalue()


def images_zip(paths: list[Path]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, arcname=p.relative_to(PROJECT_ROOT).as_posix())
    return buf.getvalue()


# -----------------------------------------------------------------------------
# Markdown renderer with local images
# -----------------------------------------------------------------------------
IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>(?:[^()\s]|\([^()]*\))+)\)")   # allows (balanced) parentheses in paths
# The backend writes the caption as an italic line directly under the image:  ![alt](path)\n*caption*
CAPTION_RE = re.compile(r"[ \t]*\r?\n[ \t]*\*(?P<cap>[^*\n]+)\*[ \t]*(?=\r?\n|$)")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)   # hides "<!-- TODO image ... -->" placeholders
FENCE_RE = re.compile(r"(```.*?```)", re.DOTALL)


def normalize_math(md: str) -> str:
    """Streamlit renders $..$ / $$..$$ but not \\(..\\) / \\[..\\]. Convert outside code fences."""
    parts = FENCE_RE.split(md)
    for i in range(0, len(parts), 2):
        p = parts[i]
        p = re.sub(r"\\\[(.+?)\\\]", lambda m: "\n$$" + m.group(1).strip() + "$$\n", p, flags=re.DOTALL)
        p = re.sub(r"\\\((.+?)\\\)", lambda m: "$" + m.group(1).strip() + "$", p, flags=re.DOTALL)
        parts[i] = p
    return "".join(parts)


def render_markdown(md: str) -> None:
    md = normalize_math(HTML_COMMENT_RE.sub("", md))
    parts = FENCE_RE.split(md)
    for index, part in enumerate(parts):
        if index % 2:
            if part.strip():
                st.markdown(part, unsafe_allow_html=False)
            continue

        cursor = 0
        for match in IMG_RE.finditer(part):
            before = part[cursor:match.start()]
            if before.strip():
                st.markdown(before, unsafe_allow_html=False)

            alt = match.group("alt").strip()
            src = match.group("src").strip()
            end = match.end()
            caption = alt or None
            cap_match = CAPTION_RE.match(part, end)
            if cap_match:
                caption = cap_match.group("cap").strip() or caption
                end = cap_match.end()

            if src.startswith(("http://", "https://")):
                st.image(src, caption=caption)
            else:
                path = (PROJECT_ROOT / src.lstrip("./")).resolve()
                try:
                    safe = path.is_relative_to(PROJECT_ROOT.resolve())
                except ValueError:
                    safe = False
                if safe and path.exists():
                    st.image(str(path), caption=caption)
                else:
                    st.warning(f"Image not found: `{src}`")
            cursor = end

        if part[cursor:].strip():
            st.markdown(part[cursor:], unsafe_allow_html=False)


# -----------------------------------------------------------------------------
# Backend run
# -----------------------------------------------------------------------------
class _LineTee:
    """Collects the backend's print() output (worker progress, rate-limit waits, image results)."""

    def __init__(self, sink: list[str], passthrough: Any) -> None:
        self.sink, self.passthrough, self.buf = sink, passthrough, ""

    def write(self, text: str) -> int:
        try:
            self.passthrough.write(text)
        except Exception:
            pass
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.sink.append(line.rstrip())
        return len(text)

    def flush(self) -> None:
        try:
            self.passthrough.flush()
        except Exception:
            pass


def _unpack(chunk: Any) -> tuple[tuple, str, Any]:
    """stream(..., stream_mode=[...], subgraphs=True) yields (namespace, mode, data)."""
    if isinstance(chunk, tuple):
        if len(chunk) == 3:
            return tuple(chunk[0]), chunk[1], chunk[2]
        if len(chunk) == 2 and isinstance(chunk[0], str):
            return (), chunk[0], chunk[1]
    return (), "updates", chunk


def _event_node(namespace: tuple, data: Any) -> str | None:
    if isinstance(data, dict):
        for name in reversed(list(data.keys())):
            if not str(name).startswith("__"):
                return str(name).split(":")[0]
    if namespace:
        for name in reversed(namespace):
            if not str(name).startswith("__"):
                return str(name).split(":")[0]
    return None


def run_graph(topic: str):
    """
    Runs the graph ONCE and yields UI events:
      {"type": "update"|"final"|"error", "node": last finished node, "state": merged state,
       "events": [log lines], "error": str, "trace": str}
    There is deliberately no fallback re-run: a retry would repeat paid LLM/Tavily/image calls.
    """
    inputs = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "sections": [],
        "final": "",
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
    }

    current: dict[str, Any] = {}   # latest top-level state ("values")
    inner: dict[str, Any] = {}     # keys produced inside the reducer subgraph (merged_md, image_specs, ...)
    live_sections: dict[int, str] = {}   # filled from "worker" updates (they arrive before the next "values")
    printed: list[str] = []
    tee = _LineTee(printed, sys.stdout)
    sent = 0

    def merged() -> dict[str, Any]:
        base = {**inner, **current} if current.get("final") else {**current, **inner}
        seen = {int(i): md for i, md in (current.get("sections") or [])}
        seen.update(live_sections)
        base["sections"] = sorted(seen.items())
        return base

    try:
        module = load_backend()
        with redirect_stdout(tee):
            for chunk in module.app.stream(
                inputs,
                config={"max_concurrency": MAX_CONCURRENCY},
                stream_mode=["updates", "values"],
                subgraphs=True,
            ):
                namespace, mode, data = _unpack(chunk)
                msgs: list[str] = []
                node: str | None = None

                if mode == "updates" and isinstance(data, dict):
                    node = _event_node(namespace, data)
                    if node:
                        msgs.append(f"➡️ {stage_label(node)} finished")
                    for name, upd in data.items():
                        if str(name).startswith("__") or not isinstance(upd, dict):
                            continue
                        if namespace:
                            inner.update(upd)
                        else:
                            # "updates" arrives before the matching "values" chunk, so apply it now
                            current.update({k: v for k, v in upd.items() if k != "sections"})
                            for item in upd.get("sections") or []:
                                live_sections[int(item[0])] = item[1]
                elif mode == "values" and not namespace and isinstance(data, dict):
                    current = dict(data)
                else:
                    continue

                msgs.extend(printed[sent:])
                sent = len(printed)
                yield {"type": "update", "node": node, "state": merged(), "events": msgs}
    except Exception as exc:
        yield {
            "type": "error",
            "node": None,
            "state": merged(),
            "events": printed[sent:] + [f"❌ {type(exc).__name__}: {exc}"],
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(),
        }
        return

    yield {"type": "final", "node": None, "state": merged(), "events": printed[sent:] + ["✅ completed"]}


# -----------------------------------------------------------------------------
# UI state
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic Blog Writer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "last_out" not in st.session_state:
    st.session_state.last_out = None
if "logs" not in st.session_state:
    st.session_state.logs = []

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("✍️ Blog Writing Agent")
st.sidebar.caption("LangGraph · Groq · Tavily · Hugging Face Images")

with st.sidebar:
    st.subheader("Generate Technical Blog")

    st.caption(
        "Create technical, developer-focused blogs using AI, "
        "ML, LLMs, RAG, LangGraph, Python, APIs, system design, and more."
    )

    topic = st.text_area(
        "Technical Topic",
        placeholder=(
            "Example:\n"
            "How RAG works with LangGraph\n\n"
            "or\n"
            "Designing a Multi-Agent AI System"
        ),
        height=140,
        help=(
            "Best results come from technical topics related to "
            "AI, ML, software engineering, Python, LLMs, "
            "RAG, agents, APIs, cloud, or system design."
        ),
    )

    st.caption(
        "💡 Best results: AI • ML • LLMs • RAG • Agents • Python • "
        "LangGraph • APIs • System Design • Cloud"
    )

    run_btn = st.button(
        "🚀 Generate Technical Blog",
        type="primary",
    )

    st.divider()
    st.subheader("Past Blogs")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved .md blogs found in the project folder.")
    else:
        labels: list[str] = []
        mapping: dict[str, Path] = {}
        for p in past_files:
            try:
                title = extract_title(read_text(p), p.stem)
            except Exception:
                title = p.stem
            label = f"{title} · {p.name}"
            labels.append(label)
            mapping[label] = p

        selected = st.selectbox("Load a saved blog", labels, index=0)
        selected_path = mapping[selected]

        if st.button("📂 Load Selected"):
            md_text = read_text(selected_path)
            st.session_state.last_out = {
                "topic": extract_title(md_text, selected_path.stem),
                "mode": "loaded",
                "needs_research": False,
                "queries": [],
                "evidence": [],
                "plan": None,
                "sections": [],
                "merged_md": "",
                "md_with_placeholders": "",
                "image_specs": [],
                "final": md_text,
                "_source_file": selected_path.name,
            }
            st.session_state.logs = [f"📂 Loaded {selected_path.name}"]
            st.rerun()

    st.divider()
    st.caption(f"Backend module: `{BACKEND_MODULE}.py`")
    st.caption("Images are controlled by `GENERATE_IMAGES` in `.env`.")


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("Blog Writing Agent")
st.caption("Router → Research → Planner → Parallel Writers → Merge → Image Planning → Image Generation")


# -----------------------------------------------------------------------------
# Run flow
# -----------------------------------------------------------------------------
try:
    load_backend()
except Exception as exc:
    st.error(f"Could not import backend module `{BACKEND_MODULE}` from `{PROJECT_ROOT}`.")
    st.code(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}", language="text")
    st.stop()

if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic first.")
        st.stop()

    st.session_state.logs = []
    current_state: dict[str, Any] = {}
    final_error: str | None = None
    error_trace: str | None = None
    last_node: str | None = None

    status = st.status("Running LangGraph…", expanded=True)
    node_placeholder = st.empty()
    metrics_placeholder = st.empty()

    for event in run_graph(topic):
        state = event.get("state", {})
        current_state = state or current_state
        if event.get("node"):
            last_node = event["node"]

        if event.get("type") != "error":
            stage = running_stage(last_node, current_state)
            done_w, total_w = count_workers(current_state)
            suffix = f" ({done_w}/{total_w} sections)" if stage == "worker" and total_w else ""
            completed = f" · Last completed: **{stage_label(last_node)}**" if last_node else ""
            node_placeholder.info(f"Current stage: **{stage_label(stage)}**{suffix}{completed}")

        for msg in event.get("events", []):
            st.session_state.logs.append(msg)
            status.write(msg)

        sections_done, sections_total = count_workers(current_state)
        evidence_count = len(current_state.get("evidence") or [])
        image_count = len(current_state.get("image_specs") or [])
        mode = current_state.get("mode") or "—"

        m1, m2, m3, m4 = metrics_placeholder.columns(4)
        m1.metric("Mode", str(mode))
        m2.metric("Evidence", evidence_count)
        m3.metric("Sections", f"{sections_done}/{sections_total}" if sections_total else "—")
        m4.metric("Images planned", image_count)

        if event.get("type") == "error":
            final_error = event.get("error")
            error_trace = event.get("trace")
            break

        if event.get("type") == "final":
            break

    st.session_state.last_out = current_state

    if final_error:
        status.update(label="❌ Run failed", state="error", expanded=True)
        st.error(final_error)
        if error_trace:
            with st.expander("Traceback"):
                st.code(error_trace, language="text")
        raw_plan = model_to_dict(current_state.get("plan")) or {}
        title = raw_plan.get("blog_title")
        if title:
            raw_path = PROJECT_ROOT / f"{safe_slug(title)}_raw.md"
            if raw_path.exists():
                st.info(f"Raw draft was preserved: `{raw_path.name}`")
    else:
        status.update(label="✅ Blog generated", state="complete", expanded=False)


# -----------------------------------------------------------------------------
# Main content tabs
# -----------------------------------------------------------------------------
out = st.session_state.last_out

tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🖼️ Images", "🧾 Logs"]
)

if not out:
    with tab_plan:
        st.info("Enter a topic in the sidebar and click **Generate Blog**.")

    with tab_logs:
        st.info("No run logs yet.")

else:
    plan = model_to_dict(out.get("plan")) or {}
    evidence = [model_to_dict(x) for x in (out.get("evidence") or [])]
    specs = [model_to_dict(x) for x in (out.get("image_specs") or [])]
    final_md = out.get("final") or ""

    # -------------------------------------------------------------------------
    # Plan tab
    # -------------------------------------------------------------------------
    with tab_plan:
        st.subheader("Content Plan")

        if not plan:
            st.info("No plan object is stored for this saved blog. Generate a new blog to see the planner output.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Blog kind", str(plan.get("blog_kind", "explainer")))
            c2.metric("Audience", str(plan.get("audience", "—")))
            c3.metric("Tone", str(plan.get("tone", "—")))

            st.markdown(f"### {plan.get('blog_title', 'Untitled')}")

            constraints = plan.get("constraints") or []
            if constraints:
                st.write("**Constraints**")
                st.write(" · ".join(str(x) for x in constraints))

            tasks = plan.get("tasks") or []
            if tasks:
                st.write("**Sections**")
                for task in sorted(tasks, key=lambda x: x.get("id", 0)):
                    title = task.get("title", "Untitled section")
                    section_type = task.get("section_type", "core")
                    words = task.get("target_words", "—")

                    with st.expander(f"#{task.get('id', '?')} · {title}  —  {section_type}"):
                        st.write(f"**Goal:** {task.get('goal', '—')}")
                        st.write(f"**Target words:** {words}")

                        bullets = task.get("bullets") or []
                        for bullet in bullets:
                            st.write(f"- {bullet}")

                        flags = []
                        if task.get("required_research"):
                            flags.append("research")
                        if task.get("required_citation"):
                            flags.append("citations")
                        if task.get("required_code"):
                            flags.append("code")
                        if flags:
                            st.caption("Required: " + " · ".join(flags))

                        tags = task.get("tags") or []
                        if tags:
                            st.caption("Tags: " + ", ".join(tags))
            else:
                st.info("No tasks found in the plan.")

    # -------------------------------------------------------------------------
    # Evidence tab
    # -------------------------------------------------------------------------
    with tab_evidence:
        st.subheader("Research Evidence")

        if out.get("mode") == "loaded":
            st.info("This blog was loaded from disk. Research evidence is not saved, only the markdown.")
        elif out.get("mode") == "closed_book":
            st.info("Router selected **closed_book**, so Tavily research was skipped.")
        elif not evidence:
            st.info("No evidence returned. This can happen if Tavily returned no results or the research step produced an empty evidence pack.")
        else:
            st.caption(f"{len(evidence)} deduplicated source(s)")
            for i, item in enumerate(evidence, start=1):
                title = item.get("title") or "Untitled source"
                url = item.get("url") or ""
                source = item.get("source") or "Unknown source"
                published = item.get("published_at") or "Date unknown"
                snippet = item.get("snippet") or ""

                with st.container(border=True):
                    st.markdown(f"**{i}. {title}**")
                    st.caption(f"{source} · {published}")
                    if url:
                        st.markdown(f"[Open source]({url})")
                    if snippet:
                        st.write(snippet)

    # -------------------------------------------------------------------------
    # Preview tab
    # -------------------------------------------------------------------------
    with tab_preview:
        st.subheader("Markdown Preview")

        if not final_md:
            st.warning("No final markdown was returned.")
        else:
            render_markdown(final_md)

            blog_title = plan.get("blog_title") or extract_title(final_md, "blog")
            filename = f"{safe_slug(blog_title)}.md"

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "⬇️ Download Markdown",
                    data=final_md.encode("utf-8"),
                    file_name=filename,
                    mime="text/markdown",
                )
            with c2:
                st.download_button(
                    "📦 Download Blog Bundle",
                    data=bundle_zip(final_md, filename),
                    file_name=f"{safe_slug(blog_title)}_bundle.zip",
                    mime="application/zip",
                )

    # -------------------------------------------------------------------------
    # Images tab
    # -------------------------------------------------------------------------
    with tab_images:
        st.subheader("Images")

        if specs:
            st.write(f"**Planned images: {len(specs)}**")
            for i, spec in enumerate(specs, start=1):
                with st.expander(f"Image {i} · {spec.get('filename', 'unnamed.png')}"):
                    st.write(f"**Alt:** {spec.get('alt', '—')}")
                    st.write(f"**Caption:** {spec.get('caption', '—')}")
                    st.write(f"**Size:** {spec.get('size', '—')}")
                    st.code(spec.get("prompt", ""), language="text")
        else:
            st.info("The image planner did not request any images for this blog.")

        blog_images = referenced_images(final_md)
        todo_matches = re.findall(r"<!--\s*TODO image\s+([^:]+):\s*(.*?)\s*-->", final_md, re.DOTALL)

        if todo_matches:
            st.warning(f"{len(todo_matches)} image(s) were not generated and remain as TODO placeholders.")

        if blog_images:
            st.write(f"**Generated image files: {len(blog_images)}**")
            cols = st.columns(2)
            for idx, path in enumerate(blog_images):
                with cols[idx % 2]:
                    st.image(str(path), caption=path.name)

            st.download_button(
                "⬇️ Download Blog Images (zip)",
                data=images_zip(blog_images),
                file_name=f"{safe_slug(plan.get('blog_title') or 'blog')}_images.zip",
                mime="application/zip",
            )

    # -------------------------------------------------------------------------
    # Logs tab
    # -------------------------------------------------------------------------
    with tab_logs:
        st.subheader("Run Logs")
        logs_text = "\n".join(st.session_state.logs[-100:])
        if logs_text:
            st.text_area("Event log", value=logs_text, height=500, label_visibility="collapsed")
        else:
            st.info("No logs available for this result.")