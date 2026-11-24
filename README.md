# pep-mcp

An [MCP](https://modelcontextprotocol.io) server that gives Claude (or any MCP client) search and reading access to
**all 747 Python Enhancement Proposals**, backed by [PEP-RAG](https://github.com/by-Kaimercer/pep-rag), a fully
local retrieval system: BM25 plus contextual dense embeddings fused with Reciprocal Rank Fusion, with PEP-number query
routing. Embeddings and the optional answer model run on Ollama, so nothing leaves your machine.

Ask Claude "how does `:=` scoping work inside comprehensions?" and it can search the PEPs, open the exact section of
PEP 572, and quote the specification instead of answering from memory.

## Tools

| Tool | What it does | Needs Ollama |
|---|---|---|
| `search_peps(query, k=5, mode="hybrid_route")` | Top-k passages (~220 words each) labelled with PEP number, title, status, Python version and section. Naming a PEP in the query (`"PEP 572: ..."`) focuses the search on it. `mode` selects any PEP-RAG strategy: `hybrid_route`, `hybrid`, `bm25`, `dense_ctx`, `dense_plain`, `hybrid_route_rerank`. | For dense modes; falls back to BM25 if Ollama is down |
| `get_pep(number)` | Full header (authors, status, type, Python version, related PEPs, resolution), the peps.python.org URL, and the section list with word counts. | No |
| `get_pep_section(number, section)` | Full text of one section. Case-insensitive; a unique partial name works, and a miss returns the valid section names. | No |
| `ask_peps(question)` | PEP-RAG's end-to-end pipeline: retrieval, then a local `qwen2.5:7b` writes a 1-4 sentence answer with `[PEP N]` citations, or says it doesn't know. | Yes |

**Resource:** `pep://{number}` returns the full reStructuredText source of a PEP.

### Design choices

- **Token-efficient by default.** Tools return compact plain text rather than JSON, `get_pep` reports word counts so
  the model can decide whether a section is worth reading, and the intended flow (search, then read one section)
  avoids pulling whole 10k-word PEPs into context.
- **Loaded once.** The lifespan hook loads the chunks, both embedding matrices and the BM25 index at startup (a few seconds),
  off the event loop. Slow Ollama calls also run in worker threads, so the server stays responsive.
- **Fails soft.** If Ollama is unreachable, `search_peps` degrades to BM25 keyword search and says so;
  `ask_peps` returns a clear tool error with the fix. Bad PEP numbers and section names return errors that tell the
  model what is valid, so it can correct itself.

## Setup

1. Build PEP-RAG once (corpus clone plus about 20 minutes of embedding on a consumer GPU). Follow its
   [README](https://github.com/by-Kaimercer/pep-rag); you need `data/chunks.jsonl`, `data/emb_plain.npy`
   and `data/emb_contextual.npy`. Keeping `data_raw_peps/` enables full headers in `get_pep` and the `pep://` resource.
2. Run Ollama with the models: `ollama serve`, `ollama pull nomic-embed-text`, `ollama pull qwen2.5:7b`.
3. Install and test this server:

```bash
git clone https://github.com/by-Kaimercer/pep-mcp
cd pep-mcp
pip install -r requirements.txt
export PEP_RAG_HOME=/path/to/pep-rag   # defaults to ../pep-rag
python test_client.py
```

### Claude Code

```bash
claude mcp add pep-rag -e PEP_RAG_HOME=/path/to/pep-rag -- python /path/to/pep-mcp/server.py
```

### Claude Desktop

Add this to `claude_desktop_config.json` (Settings > Developer > Edit Config), then restart Claude Desktop:

```json
{
  "mcpServers": {
    "pep-rag": {
      "command": "python",
      "args": ["/path/to/pep-mcp/server.py"],
      "env": { "PEP_RAG_HOME": "/path/to/pep-rag" }
    }
  }
}
```

On Windows, use full paths such as `C:\\Users\\you\\pep-mcp\\server.py`, and the full path to `python.exe` if
Python is not on the PATH Claude Desktop sees.

## Example transcript

These are real outputs from `python test_client.py` (the full log is in [`test_output.txt`](test_output.txt)),
arranged as the calls Claude makes when a user asks *"Can I use `:=` inside a list comprehension in a class body?"*

**`search_peps({"query": "PEP 572: scope of assignment expression targets inside comprehensions", "k": 3})`**

```text
[1] PEP 572: Assignment Expressions (Final, Python 3.8)
Section: Syntax and semantics > Scope of the target
A further exception applies when an assignment expression occurs in a
comprehension whose containing scope is a class scope. If the rules
above were to result in the target being assigned in that class's
scope, the assignment expression is expressly invalid. This case also raises
``SyntaxError``::

class Example:
 [(j := i) for i in range(5)] # INVALID
...
```

**`get_pep({"number": 572})`**

```text
PEP: 572
Title: Assignment Expressions
Author: Chris Angelico <rosuav@gmail.com>, Tim Peters <tim.peters@gmail.com>, Guido van Rossum <guido@python.org>
Status: Final
Type: Standards Track
Created: 28-Feb-2018
Python-Version: 3.8
...
URL: https://peps.python.org/pep-0572/

Sections (words):
- Abstract (109)
- Rationale (126)
- Rationale > The importance of real code (383)
- Syntax and semantics (137)
...
```

**`get_pep_section({"number": 572, "section": "nope"})`**, an error the model can recover from:

```text
Error executing tool get_pep_section: No section matches 'nope' in PEP 572. Choose one of: Abstract; Rationale;
Rationale > The importance of real code; Syntax and semantics; Syntax and semantics > Exceptional cases; ...
```

**`ask_peps({"question": "What maximum line length does PEP 8 recommend?"})`**, answered locally in 4.3 s:

```text
PEP 8 recommends a maximum line width of 79 characters for most lines, with docstrings and comments limited to
72 characters [PEP 8].

Retrieved sources:
- PEP 8: Style Guide for Python Code / Code Lay-out > Maximum Line Length
...
```

**With Ollama stopped**, `search_peps({"query": "pattern matching", "k": 2})`:

```text
(Ollama unreachable: fell back to BM25 keyword search. Ollama is not reachable at http://localhost:1. Start it with
`ollama serve` (models: nomic-embed-text, qwen2.5:7b) and retry.)

[1] PEP 636: Structural Pattern Matching: Tutorial (Final, Python 3.10)
Section: Abstract
...
```

## Known limitation

PEP-RAG retrieves by wording, so a nickname can lose to a literal match: "walrus operator" pulls PEP 622's
*walrus patterns* ahead of PEP 572, which calls the feature an *assignment expression*. Naming the PEP in the query
(`"PEP 572: ..."`) triggers routing and fixes it, which is why the tool description tells the model about it.
