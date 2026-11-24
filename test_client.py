"""End-to-end smoke test: spawn server.py over stdio with the official MCP client, list tools, call every tool,
read a pep:// resource, then restart the server against a dead Ollama URL to check graceful failure.

Run:  python test_client.py
"""
import asyncio
import os
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = str(Path(__file__).with_name("server.py"))
failures = []


def show(label, result, max_chars=600):
    text = "\n".join(getattr(c, "text", "") for c in result.content)
    status = "ERROR" if result.isError else "ok"
    print(f"\n=== {label} [{status}] ===\n{text[:max_chars]}{' ...' if len(text) > max_chars else ''}")
    return text


async def call(session, label, tool, args, expect_error=False, must_contain=None):
    t0 = time.perf_counter()
    res = await session.call_tool(tool, args)
    text = show(f"{label} ({time.perf_counter() - t0:.1f}s)", res)
    if res.isError != expect_error or (must_contain and must_contain not in text):
        failures.append(label)
    return text


async def run(env=None):
    params = StdioServerParameters(command=sys.executable, args=[SERVER], env={**os.environ, **(env or {})})
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        yield session


async def main():
    async for s in run():
        tools = (await s.list_tools()).tools
        print("Tools:")
        for t in tools:
            print(f"  - {t.name}({', '.join(t.inputSchema.get('properties', {}))})")
        templates = (await s.list_resource_templates()).resourceTemplates
        print("Resource templates:", [t.uriTemplate for t in templates])
        if {t.name for t in tools} != {"search_peps", "get_pep", "get_pep_section", "ask_peps"}:
            failures.append("list_tools")

        await call(s, "search_peps", "search_peps", {"query": "PEP 572: scope of assignment expression targets inside comprehensions", "k": 3}, must_contain="PEP 572")
        await call(s, "get_pep", "get_pep", {"number": 572}, must_contain="Assignment Expressions")
        await call(s, "get_pep_section", "get_pep_section", {"number": 572, "section": "rationale"}, must_contain="Section: Rationale")
        await call(s, "get_pep_section (bad name)", "get_pep_section", {"number": 572, "section": "nope"}, expect_error=True)
        await call(s, "get_pep (unknown)", "get_pep", {"number": 9998}, expect_error=True)
        await call(s, "ask_peps", "ask_peps", {"question": "What maximum line length does PEP 8 recommend?"}, must_contain="[PEP 8]")

        res = await s.read_resource("pep://8")
        body = res.contents[0].text
        print(f"\n=== resource pep://8 ===\n{body[:200]} ... ({len(body)} chars)")
        if "Style Guide" not in body:
            failures.append("resource")

    print("\n--- Restarting server with OLLAMA_URL pointing at a dead port ---")
    async for s in run({"OLLAMA_URL": "http://localhost:1"}):
        await call(s, "search_peps (Ollama down)", "search_peps", {"query": "pattern matching", "k": 2}, must_contain="fell back to BM25")
        await call(s, "ask_peps (Ollama down)", "ask_peps", {"question": "What is pattern matching?"}, expect_error=True)

    print("\nRESULT:", "PASS" if not failures else f"FAIL {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
