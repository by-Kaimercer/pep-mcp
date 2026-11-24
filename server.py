"""PEP-MCP: an MCP server that exposes the PEP-RAG retrieval system (all Python PEPs) as tools for Claude.

Run:  python server.py            (stdio transport; PEP_RAG_HOME points at a built pep-rag checkout)
"""
import os
import sys
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import anyio
import requests
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import Field

PEP_RAG_HOME = Path(os.environ.get("PEP_RAG_HOME", Path(__file__).resolve().parent.parent / "pep-rag")).resolve()
sys.path.insert(0, str(PEP_RAG_HOME))

from pep_rag.answer import ask  # noqa: E402
from pep_rag.config import OLLAMA_URL, RAW_DIR  # noqa: E402
from pep_rag.index import tokenize  # noqa: E402
from pep_rag.ingest import parse_header  # noqa: E402
from pep_rag.retrieve import Retriever  # noqa: E402

OLLAMA_DOWN = (
    f"Ollama is not reachable at {OLLAMA_URL}. Start it with `ollama serve` "
    "(models: nomic-embed-text, qwen2.5:7b) and retry."
)

state: dict = {}


@asynccontextmanager
async def lifespan(_server):
    # Load chunks, both embedding matrices and the BM25 index once, off the event loop.
    state["r"] = await anyio.to_thread.run_sync(Retriever)
    yield


mcp = FastMCP(
    "pep-rag",
    instructions=(
        "Search and read the full corpus of Python Enhancement Proposals (PEPs). "
        "Prefer search_peps, then get_pep / get_pep_section to read exact text and cite it yourself; "
        "ask_peps runs a local 7B model and is slower."
    ),
    lifespan=lifespan,
    log_level="WARNING",
)


def retriever() -> Retriever:
    return state["r"]


def pep_chunks(number: int) -> list[dict]:
    r = retriever()
    if number not in r.by_pep:
        raise ToolError(f"PEP {number} is not in the corpus ({len(r.by_pep)} PEPs indexed).")
    return [r.chunks[i] for i in r.by_pep[number]]


def sections_of(chunks: list[dict]) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for c in chunks:
        out[c["section"]].append(c)
    return out


def fmt_chunk(n: int, c: dict) -> str:
    pv = f", Python {c['python_version']}" if c["python_version"] else ""
    return f"[{n}] PEP {c['pep']}: {c['title']} ({c['status']}{pv})\nSection: {c['section']}\n{c['text']}"


def is_ollama_error(e: Exception) -> bool:
    return isinstance(e, (requests.ConnectionError, requests.Timeout))


@mcp.tool()
async def search_peps(
    query: Annotated[str, Field(description="Natural-language question or keywords. Mentioning 'PEP 572' focuses the search on that PEP.")],
    k: Annotated[int, Field(ge=1, le=20, description="Number of passages to return.")] = 5,
    mode: Annotated[
        Literal["hybrid_route", "hybrid", "bm25", "dense_ctx", "dense_plain", "hybrid_route_rerank"],
        Field(description="Retrieval strategy. hybrid_route (default) = BM25 + contextual embeddings fused with RRF, "
              "restricted to any PEP named in the query. hybrid_route_rerank adds a slow local-LLM rerank."),
    ] = "hybrid_route",
) -> str:
    """Search all Python Enhancement Proposals and return the most relevant passages (~220 words each),
    each labelled with PEP number, title, status, Python version and section. Use this first for any question
    about Python language design, the standard library, packaging or governance."""
    r = retriever()
    note = ""
    try:
        hits = await anyio.to_thread.run_sync(lambda: r.search(query, mode=mode, k=k))
    except Exception as e:
        if not is_ollama_error(e):
            raise
        # Graceful degradation: BM25 needs no model, so keyword search still works without Ollama.
        idx = r._top(r.bm25.scores(tokenize(query)), k)
        hits = [r.chunks[i] for i in idx]
        note = f"(Ollama unreachable: fell back to BM25 keyword search. {OLLAMA_DOWN})\n\n"
    return note + "\n\n".join(fmt_chunk(n + 1, c) for n, c in enumerate(hits))


@mcp.tool()
def get_pep(number: Annotated[int, Field(ge=0, le=9999, description="PEP number, e.g. 572.")]) -> str:
    """Get a PEP's header metadata (title, status, type, authors, Python version, related PEPs) and its list of
    sections with word counts. Use it to see what a PEP covers before reading a section with get_pep_section."""
    chunks = pep_chunks(number)
    raw = RAW_DIR / f"pep-{number:04d}.rst"
    if raw.exists():
        meta, _ = parse_header(raw.read_text(encoding="utf-8", errors="replace").splitlines())
    else:
        c = chunks[0]
        meta = {"PEP": number, "Title": c["title"], "Status": c["status"], "Type": c["type"],
                "Created": c["created"], "Python-Version": c["python_version"]}
    lines = [f"{k}: {v}" for k, v in meta.items() if v]
    lines.append(f"URL: https://peps.python.org/pep-{number:04d}/")
    lines.append("\nSections (words):")
    for name, cs in sections_of(chunks).items():
        lines.append(f"- {name} ({sum(len(c['text'].split()) for c in cs)})")
    return "\n".join(lines)


@mcp.tool()
def get_pep_section(
    number: Annotated[int, Field(ge=0, le=9999, description="PEP number, e.g. 572.")],
    section: Annotated[str, Field(description="Section name as listed by get_pep, e.g. 'Rationale' or "
                                  "'Specification > Scope of the target'. Case-insensitive; a unique partial match works.")],
) -> str:
    """Return the full text of one section of a PEP. Subsections are separate sections (see get_pep)."""
    secs = sections_of(pep_chunks(number))
    want = section.strip().lower()
    exact = [s for s in secs if s.lower() == want or s.lower().split(" > ")[-1] == want]
    matches = exact or [s for s in secs if want in s.lower()]
    if len(matches) != 1:
        problem = "No section matches" if not matches else "Several sections match"
        options = matches or list(secs)
        raise ToolError(f"{problem} '{section}' in PEP {number}. Choose one of: " + "; ".join(options))
    name = matches[0]
    c = secs[name][0]
    return f"PEP {number}: {c['title']}\nSection: {name}\n\n" + "\n\n".join(x["text"] for x in secs[name])


@mcp.tool()
async def ask_peps(
    question: Annotated[str, Field(description="A question about Python answerable from the PEPs.")],
) -> str:
    """Answer a question with PEP-RAG's full pipeline: hybrid retrieval, then a local qwen2.5:7b model writes a short
    grounded answer with [PEP N] citations, or says it doesn't know. Slower (seconds) than search_peps; use it when
    you want PEP-RAG's own cited answer rather than raw passages."""
    try:
        out = await anyio.to_thread.run_sync(lambda: ask(question, retriever()))
    except Exception as e:
        if is_ollama_error(e):
            raise ToolError(OLLAMA_DOWN) from e
        raise
    srcs = "\n".join(f"- PEP {c['pep']}: {c['title']} / {c['section']}" for c in out["sources"])
    return f"{out['answer']}\n\nRetrieved sources:\n{srcs}"


@mcp.resource("pep://{number}", mime_type="text/x-rst")
def pep_resource(number: str) -> str:
    """Full source text of a PEP (reStructuredText)."""
    n = int(number)
    raw = RAW_DIR / f"pep-{n:04d}.rst"
    if raw.exists():
        return raw.read_text(encoding="utf-8", errors="replace")
    return "\n\n".join(f"{name}\n\n" + "\n\n".join(c["text"] for c in cs) for name, cs in sections_of(pep_chunks(n)).items())


if __name__ == "__main__":
    mcp.run()
