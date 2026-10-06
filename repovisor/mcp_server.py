"""Read-only MCP server. Run: python -m repovisor.mcp_server  (stdio transport)
Claude Desktop config: {"mcpServers": {"repovisor": {"command": "python", "args": ["-m", "repovisor.mcp_server"]}}}
Write/action tools are intentionally absent; add them in a separate server."""
from mcp.server.fastmcp import FastMCP
from . import core

mcp = FastMCP("repovisor")


@mcp.tool()
def repo_metadata(repo_id: str) -> dict:
    """Source, commit, and size stats for an indexed repository."""
    m = core.load(repo_id)[0]
    return dict(source=m["source"], commit=m["commit"], files=len(m["files"]), symbols=len(m["symbols"]))


@mcp.tool()
def search_code(repo_id: str, query: str, k: int = 6) -> list[dict]:
    """Semantic code/doc search. Returns file, line range, and snippet."""
    return [dict(file=h["file"], lines=f"{h['start']}-{h['end']}", text=h["text"][:1500])
            for h in core.search(repo_id, query, min(k, 10))]


@mcp.tool()
def get_file(repo_id: str, path: str, start: int = 1, end: int = 400) -> str:
    """Read a line range from a file in the repository."""
    return core.read_file(repo_id, path, start, end)


@mcp.tool()
def find_symbol(repo_id: str, name: str) -> list[dict]:
    """Find classes/functions by (partial) name."""
    return core.find_symbol(repo_id, name)


@mcp.tool()
def dependencies(repo_id: str, path: str) -> dict:
    """Files this file imports and files that import it."""
    return core.deps(repo_id, path)


@mcp.tool()
def who_calls(repo_id: str, function_name: str) -> list:
    """Functions (and their files) that call the given function name."""
    return core.callers(repo_id, function_name)


@mcp.tool()
def commit_history(repo_id: str, path: str | None = None, n: int = 10) -> list[str]:
    """Recent commits (hash|author|date|subject), optionally for one file."""
    return core.history(repo_id, path, min(n, 30))


if __name__ == "__main__":
    mcp.run()
