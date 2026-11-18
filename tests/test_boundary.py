"""The library / bot boundary (4.0): tri never imports the bot, never reads the
environment and never opens a network connection; the bot imports tri only
through tri.api."""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
NETWORK = {"urllib.request", "http.client", "socket", "requests", "aiohttp", "httpx", "discord", "ssl"}


def _files(pkg):
    return sorted(p for p in (ROOT / pkg).rglob("*.py") if "__pycache__" not in p.parts)


def _imports(path):
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Import):
            for a in n.names:
                yield n.lineno, a.name, None
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            for a in n.names:
                yield n.lineno, n.module, a.name


def test_tri_does_not_import_bot_or_network():
    bad = []
    for p in _files("tri"):
        for ln, mod, _ in _imports(p):
            top = mod.split(".")[0]
            if top == "bot" or mod in NETWORK or top in NETWORK:
                bad.append(f"{p.relative_to(ROOT)}:{ln} imports {mod}")
    assert not bad, bad


def test_tri_reads_no_environment():
    bad = []
    for p in _files("tri"):
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n, ast.Attribute) and n.attr in ("environ", "getenv", "putenv") and \
                    isinstance(n.value, ast.Name) and n.value.id == "os":
                bad.append(f"{p.relative_to(ROOT)}:{n.lineno} os.{n.attr}")
    assert not bad, bad


def test_bot_imports_only_the_public_api():
    bad = []
    for p in _files("bot"):
        for ln, mod, name in _imports(p):
            if mod == "tri" or mod.startswith("tri."):
                if mod != "tri.api":
                    bad.append(f"{p.relative_to(ROOT)}:{ln} imports {mod}" + (f".{name}" if name else ""))
    assert not bad, bad


def test_api_names_resolve():
    import tri.api as api
    missing = [n for n in api.__all__ if not hasattr(api, n)]
    assert not missing, missing
