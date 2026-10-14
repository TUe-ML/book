#!/usr/bin/env python3
"""Turn a Jupyter Book 2 (MyST) HTML build into a copy that opens from disk.

The MyST book theme serves pages with root-absolute links and a JavaScript
app (ES modules + fetch) that browsers block on file:// URLs. This script
keeps the pre-rendered HTML, drops the module scripts, makes every link
relative, regenerates the sidebar table of contents, downloads CDN
stylesheets (KaTeX, Font Awesome, ...) into the output, and adds a small
classic script for the sidebar folders, theme toggle, mobile menu, code
copy buttons and search.

Usage:
    jupyter book build --html
    python scripts/make_offline.py [_build/html] [_build/offline]

Then open _build/offline/index.html in a browser.
"""

import html
import json
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ASSET_DIR = "_offline"


# --------------------------------------------------------------------------
# CDN stylesheets


def fetch(url, cache):
    """Download url once into the cache directory and return the bytes."""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", url)
    path = cache / name
    if not path.exists():
        print(f"  downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "make_offline"})
        with urllib.request.urlopen(req, timeout=60) as r:
            path.write_bytes(r.read())
    return path.read_bytes()


def local_path_for(url):
    """Map a CDN URL to a path below _offline/vendor/."""
    p = urllib.parse.urlsplit(url)
    return f"{ASSET_DIR}/vendor/{p.netloc}{p.path}"


def vendor_stylesheet(url, out, cache):
    """Copy a remote stylesheet and the files its url(...) entries point to."""
    target = out / local_path_for(url)
    if target.exists():
        return
    css = fetch(url, cache).decode("utf-8")
    for ref in set(re.findall(r"url\(\s*['\"]?([^'\")]+)['\"]?\s*\)", css)):
        if ref.startswith(("data:", "#")):
            continue
        dep_url = urllib.parse.urljoin(url, ref)
        dep_url = urllib.parse.urlsplit(dep_url)._replace(query="", fragment="").geturl()
        dep_target = out / local_path_for(dep_url)
        if not dep_target.exists():
            dep_target.parent.mkdir(parents=True, exist_ok=True)
            dep_target.write_bytes(fetch(dep_url, cache))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(css, encoding="utf-8")


# --------------------------------------------------------------------------
# Links


class Linker:
    def __init__(self, out, slugs, index_slug):
        self.out = out
        self.slugs = set(slugs)
        self.index_slug = index_slug

    def target(self, url):
        """Map a root-absolute site URL to a path relative to the site root."""
        path, sep, frag = url.partition("#")
        frag = sep + frag
        query = ""
        if "?" in path:
            path, query = path.split("?", 1)
            query = "?" + query
        slug = path.strip("/")
        if slug == "" or slug == self.index_slug:
            return "index.html" + frag
        if slug in self.slugs or (self.out / slug / "index.html").exists():
            return f"{slug}/index.html{frag}"
        return path.lstrip("/") + query + frag

    def rewrite(self, page_html, prefix):
        def attr(m):
            name, quote, url = m.group(1), m.group(2), m.group(3)
            if url.startswith("//"):
                return m.group(0)
            return f"{name}={quote}{prefix}{self.target(html.unescape(url))}{quote}"

        return re.sub(r'\b(href|src|poster)=(["\'])(/[^"\']*)\2', attr, page_html)


# --------------------------------------------------------------------------
# Sidebar table of contents

LINK_CLS = (
    "block break-words focus:outline outline-myst-focus-outline outline-2 rounded"
)
ITEM_CLS = LINK_CLS + " myst-toc-item p-2 my-1 rounded-lg hover:bg-myst-surface"
ACTIVE_CLS = " myst-toc-item-exact bg-myst-active-bg active"
CHEVRON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"'
    ' aria-hidden="true" class="transition-transform duration-300'
    ' group-data-[state=open]:rotate-90 text-myst-text-secondary" height="1.5rem"'
    ' width="1.5rem"><path fill-rule="evenodd" d="M16.28 11.47a.75.75 0 0 1 0 1.06l-7.5'
    " 7.5a.75.75 0 0 1-1.06-1.06L14.69 12 7.72 5.03a.75.75 0 0 1 1.06-1.06l7.5 7.5Z\""
    ' clip-rule="evenodd"></path></svg>'
)


def build_tree(pages):
    """Nest the flat page list (with levels) into (page, children) pairs."""
    root = []
    stack = [(0, root)]
    for page in pages:
        while stack[-1][0] >= page["level"]:
            stack.pop()
        node = (page, [])
        stack[-1][1].append(node)
        stack.append((page["level"], node[1]))
    return root


def contains(node, slug):
    page, children = node
    return page["slug"] == slug or any(contains(c, slug) for c in children)


def render_toc(tree, title, current, prefix, linker):
    counter = [0]

    def link(slug, text, cls):
        href = prefix + linker.target("/" + slug)
        cur = ' aria-current="page"' if slug == current else ""
        return (
            f'<a title="{html.escape(text)}"{cur} class="{cls}" href="{href}">'
            f"{html.escape(text)}</a>"
        )

    def nodes(items):
        parts = []
        for page, children in items:
            slug, text = page["slug"], page["title"]
            active = ACTIVE_CLS if slug == current else ""
            if not children:
                parts.append(link(slug, text, ITEM_CLS + active))
                continue
            counter[0] += 1
            cid = f"offline-toc-{counter[0]}"
            is_open = contains((page, children), current)
            state = "open" if is_open else "closed"
            parts.append(
                f'<div data-state="{state}" class="w-full">'
                '<div class="myst-toc-item flex flex-row w-full gap-2 pl-2 my-1 text-left'
                f' rounded-lg outline-none hover:bg-myst-surface{active}">'
                + link(slug, text, LINK_CLS + " py-2 grow")
                + '<button class="self-stretch flex items-center flex-none px-1 rounded-l-md'
                " group hover:bg-myst-surface focus-visible:outline outline-myst-focus-outline"
                f' outline-2" aria-label="Open Folder" type="button" aria-controls="{cid}"'
                f' aria-expanded="{str(is_open).lower()}" data-state="{state}">{CHEVRON}</button>'
                "</div>"
                f'<div data-state="{state}" id="{cid}"{"" if is_open else " hidden"}'
                ' class="pl-3 pr-[2px] collapsible-content" style="animation:none">'
                + nodes(children)
                + "</div></div>"
            )
        return "".join(parts)

    home_active = ACTIVE_CLS if current == "" else ""
    home = (
        f'<a title="{html.escape(title)}" class="{ITEM_CLS} font-bold{home_active}"'
        f' href="{prefix}index.html">{html.escape(title)}</a>'
    )
    return '<div class="myst-toc w-full px-1 text-myst-text">' + home + nodes(tree) + "</div>"


# --------------------------------------------------------------------------
# Notebook outputs the theme renders in the browser ("Loading..." in the HTML)

LOADING = '<div><div class="p-2.5">Loading...</div></div>'
CLIENT_MIMES = ("text/html", "image/svg+xml", "application/vnd.")


def collect_outputs(node, found):
    if isinstance(node, dict):
        if node.get("type") == "outputs" and "id" in node:
            found[node["id"]] = node
            return
        for v in node.values():
            collect_outputs(v, found)
    elif isinstance(node, list):
        for v in node:
            collect_outputs(v, found)


def render_output(data, out, prefix):
    """Static HTML for one Jupyter output bundle."""

    def body(mime):
        item = data[mime]
        if "content" in item:
            return item["content"]
        return (out / item["path"].lstrip("/")).read_text(encoding="utf-8")

    def img(mime):
        item = data[mime]
        if "path" in item:
            return f'<img src="{prefix}{item["path"].lstrip("/")}" alt=""/>'
        return f'<img src="data:{mime};base64,{item["content"]}" alt=""/>'

    interactive = any(k.startswith("application/vnd.") for k in data)
    if interactive:
        # Widgets and plotly need a kernel or a JS library; use the saved image.
        if "image/png" in data:
            return img("image/png")
        if "text/html" in data:
            return body("text/html")
        return ""
    if "text/html" in data:
        return f'<div class="offline-html-output">{body("text/html")}</div>'
    if "image/svg+xml" in data:
        return img("image/svg+xml")
    if "image/png" in data:
        return img("image/png")
    if "text/plain" in data:
        return f"<pre>{html.escape(body('text/plain'))}</pre>"
    return ""


def fill_outputs(src, outputs, out, prefix, rel):
    for oid, node in outputs.items():
        start = src.find(f'data-mdast-node-id="{oid}"')
        if start < 0 or src.find(LOADING, start) < 0:
            continue
        pending = [
            (c.get("jupyter_data") or {}).get("data") or {}
            for c in node.get("children", [])
        ]
        pending = [d for d in pending if any(k.startswith(CLIENT_MIMES) for k in d)]
        pos = src.find(">", start) + 1
        for data in pending:
            i = src.find(LOADING, pos)
            nxt = src.find("data-mdast-node-id=", pos)
            if i < 0 or (0 <= nxt < i):
                print(f"  warning: {rel}: output {oid} has fewer placeholders than expected")
                break
            rendered = f"<div>{render_output(data, out, prefix)}</div>"
            src = src[:i] + rendered + src[i + len(LOADING):]
            pos = i + len(rendered)
    if LOADING in src:
        print(f"  warning: {rel}: {src.count(LOADING)} output(s) left as 'Loading...'")
    return src


# --------------------------------------------------------------------------
# Pages

REMIX_CONTEXT = re.compile(r"<script>\s*window\.__remixContext = (\{.*?\});\s*</script>", re.S)
DROP_SCRIPTS = (
    "window.__remixContext",  # page data for the JS app
    "__remixRouteModules",  # module entry point
    "endsWith('/index.html')",  # redirects .../index.html to .../ (breaks file://)
    "sessionStorage",  # scroll restoration for client-side routing
)


def process_page(src, rel, linker, tree, title, out):
    depth = rel.count("/")
    prefix = "../" * depth
    current = "" if rel == "index.html" else rel.rsplit("/", 1)[0]

    outputs = {}
    collect_outputs(json.loads(REMIX_CONTEXT.search(src).group(1))["state"]["loaderData"], outputs)

    # Remote stylesheets -> local copies.
    def css_link(m):
        tag = m.group(0)
        url = re.search(r'href="(https?://[^"]+)"', tag).group(1)
        vendor_stylesheet(url, out, linker.cache)
        return f'<link rel="stylesheet" href="{prefix}{local_path_for(url)}"/>'

    src = re.sub(r'<link rel="stylesheet" href="https?://[^"]+"[^>]*>', css_link, src)

    # Drop the JavaScript app.
    src = re.sub(r'<link rel="modulepreload"[^>]*>', "", src)

    def script(m):
        body = m.group(0)
        if 'type="module"' in body[:40] or any(k in body for k in DROP_SCRIPTS):
            return ""
        return body

    src = re.sub(r"<script\b[^>]*>.*?</script>", script, src, flags=re.S)

    # Full table of contents instead of the half-rendered one.
    toc = render_toc(tree, title, current, prefix, linker)
    src = re.sub(
        r'<div class="myst-toc w-full px-1 text-myst-text">.*?(?=</nav>)',
        lambda m: toc,
        src,
        count=1,
        flags=re.S,
    )

    src = linker.rewrite(src, prefix)
    src = fill_outputs(src, outputs, out, prefix, rel)

    # Offline helpers.
    tags = (
        f'<link rel="stylesheet" href="{prefix}{ASSET_DIR}/offline.css"/>'
        f'<script>window.OFFLINE_ROOT={json.dumps(prefix)};</script>'
        f'<script defer src="{prefix}{ASSET_DIR}/offline.js"></script>'
    )
    return src.replace("</head>", tags + "</head>", 1)


# --------------------------------------------------------------------------


def main():
    src_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "_build/html").resolve()
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "_build/offline").resolve()
    cache = src_dir.parent / ".offline-cache"
    cache.mkdir(exist_ok=True)

    if not (src_dir / "index.html").exists():
        sys.exit(f"{src_dir} has no index.html; run `jupyter book build --html` first")

    print(f"copying {src_dir} -> {out}")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src_dir, out)

    first = (out / "index.html").read_text(encoding="utf-8")
    ctx = json.loads(REMIX_CONTEXT.search(first).group(1))
    project = ctx["state"]["loaderData"]["root"]["config"]["projects"][0]
    pages = project["pages"]
    title = project.get("title") or "Home"

    linker = Linker(out, [p["slug"] for p in pages], project.get("index"))
    linker.cache = cache
    tree = build_tree(pages)

    count = 0
    for path in sorted(out.rglob("index.html")):
        rel = path.relative_to(out).as_posix()
        text = path.read_text(encoding="utf-8")
        if "__remixContext" not in text:
            continue
        path.write_text(process_page(text, rel, linker, tree, title, out), encoding="utf-8")
        count += 1
    print(f"processed {count} pages")

    # Search index as a classic script (fetch() is blocked on file://).
    assets = out / ASSET_DIR
    assets.mkdir(exist_ok=True)
    search = out / "myst.search.json"
    records = json.loads(search.read_text(encoding="utf-8"))["records"] if search.exists() else []
    for r in records:
        r["url"] = linker.target(r["url"])
    (assets / "search-index.js").write_text(
        "window.OFFLINE_SEARCH=" + json.dumps(records, ensure_ascii=False) + ";",
        encoding="utf-8",
    )

    here = Path(__file__).resolve().parent / "offline_assets"
    for name in ("offline.js", "offline.css"):
        shutil.copy(here / name, assets / name)

    print(f"done: open {out / 'index.html'}")


if __name__ == "__main__":
    main()
