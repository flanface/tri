"""4.0.2: the bot draws nothing; the graphs are a static site (bot.site)."""
import ast
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
GRAPHICS = {"matplotlib", "seaborn", "PIL"}


def test_bot_imports_no_graphics_library():
    bad = []
    for p in sorted((ROOT / "bot").rglob("*.py")):
        for n in ast.walk(ast.parse(p.read_text())):
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else \
                [n.module] if isinstance(n, ast.ImportFrom) and n.module else []
            bad += [f"{p.relative_to(ROOT)}:{n.lineno} {m}" for m in mods if m.split(".")[0] in GRAPHICS]
    assert not bad, bad


def test_bot_loads_without_matplotlib():
    code = ("import sys; import bot.discord_bot, bot.commands, bot.inline, bot.insight, bot.site; "
            "print(any(m.split('.')[0] in ('matplotlib', 'seaborn') for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "False"


def test_no_plot_command_or_inline_worth():
    from bot import inline
    from bot.commands import MODULES
    assert "plot" not in {m.__name__.rsplit(".", 1)[1] for m in MODULES}
    assert inline.parse("worth of Blinding Dawn") is None
    assert inline.parse("a perp b").kind == "perp"
    assert "worth of" not in inline.HELP


def test_index_has_one_lazy_tab_per_graph():
    from bot import site
    page = site._index({"run": "x", "reviews": 3})
    for k, title, _ in site.TABS:
        assert f'data-tab="{k}"' in page and f'data-src="{k}.html"' in page and title in page
    assert not re.search(r'<iframe[^>]* src=', page)          # nothing loads before its tab opens


def test_site_is_committed_with_every_tab():
    d = ROOT / "site"
    from bot import site
    assert (d / ".nojekyll").exists() and (d / "index.html").exists()
    for k, _, _ in site.TABS:
        assert (d / f"{k}.html").stat().st_size > 1000
