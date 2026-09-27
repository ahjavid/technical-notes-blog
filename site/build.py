#!/usr/bin/env python3
"""Static site builder for the Technical Notes blog.

    python site/build.py            build into _site/ and check every internal link
    python site/build.py --serve    build, serve at http://localhost:8000, rebuild on change
    python site/build.py --drafts   also build posts marked `draft: true`

Content is Markdown with YAML front matter (see CONTRIBUTING.md):

    posts/<slug>/index.md   a post
    posts/<slug>/*.md       appendices, published next to the post
    pages/<name>.md         standalone pages, published at /<name>/

Images, CSV and JSON files inside a post folder are copied as-is. Source code is
not: Python packages, *.py files and tests stay in the repository, and Markdown
links to them are rewritten to point at GitHub, so the same link works on the
site and when reading the Markdown on GitHub.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import html
import http.server
import math
import os
import posixpath
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from markupsafe import Markup
from markdown_it import MarkdownIt
from markdown_it.common.utils import unescapeAll
from markdown_it.token import Token
from mdit_py_plugins.anchors import anchors_plugin
from mdit_py_plugins.footnote import footnote_plugin
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
FRONT_MATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n", re.S)
WORDS_PER_MINUTE = 230
REQUIRED_POST_FIELDS = ("title", "description", "date", "tags")
# Files that are never published, even when they sit inside a post folder.
ALWAYS_EXCLUDED = ("*.py", "*.pyc", "*.pyo", "pyproject.toml", "setup.cfg",
                   "requirements*.txt", ".DS_Store", "Thumbs.db")
CODE_LANG_LABELS = {"py": "python", "sh": "bash", "shell": "bash", "yml": "yaml"}
PLAIN_LANGS = {"", "text", "txt", "plain", "plaintext"}


class BuildError(Exception):
    pass


def slugify(text: str) -> str:
    """URL slug for topics: 'LLM-as-a-Judge' -> 'llm-as-a-judge'."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def heading_slug(text: str) -> str:
    """Heading id compatible with GitHub, so `file.md#section` links work in both places."""
    return re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")


def human_date(value: dt.date) -> str:
    return f"{value:%B} {value.day}, {value.year}"


def as_date(value, source: Path, key: str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError as exc:
        raise BuildError(f"{source}: `{key}` must be a YYYY-MM-DD date, got {value!r}") from exc


def split_front_matter(text: str) -> tuple[dict, str]:
    match = FRONT_MATTER.match(text)
    if not match:
        return {}, text
    return (yaml.safe_load(match.group(1)) or {}), text[match.end():]


def reading_minutes(markdown: str) -> int:
    prose = re.sub(r"^(```|~~~).*?^\1", "", markdown, flags=re.S | re.M)
    prose = re.sub(r"<[^>]+>|[#>*_`|\-]", " ", prose)
    return max(1, math.ceil(len(prose.split()) / WORDS_PER_MINUTE))


def rel_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


@dataclass
class Page:
    source: Path
    kind: str  # "post", "doc" (appendix inside a post folder) or "page"
    url: str  # path below the site root: "", "about/", "posts/x/", "posts/x/notes.html"
    meta: dict
    body: str
    title: str = ""
    description: str = ""
    content: Markup = Markup("")
    toc: list[dict] = field(default_factory=list)
    post: Page | None = None  # the post a doc belongs to
    docs: list[Page] = field(default_factory=list)  # a post's appendices
    prev: Page | None = None
    next: Page | None = None

    @property
    def rel(self) -> str:
        return rel_path(self.source)

    @property
    def date(self) -> dt.date | None:
        return self.meta.get("date")

    @property
    def updated(self) -> dt.date | None:
        return self.meta.get("updated")

    @property
    def folder(self) -> str:
        return rel_path(self.source.parent)


class Site:
    def __init__(self, out: Path, include_drafts: bool = False):
        self.config = yaml.safe_load((SITE / "config.yml").read_text(encoding="utf-8"))
        self.out = out
        self.include_drafts = include_drafts
        self.base = "/" + self.config["base_path"].strip("/") + "/" if self.config["base_path"].strip("/") else "/"
        self.origin = "{0.scheme}://{0.netloc}".format(urlsplit(self.config["url"]))
        self.exclude = list(ALWAYS_EXCLUDED) + list(self.config.get("exclude", []))
        self.pages: list[Page] = []
        self.posts: list[Page] = []
        self.assets: dict[str, str] = {}  # repo-relative source -> site path
        self.url_of: dict[str, str] = {}  # repo-relative source -> site path (pages and assets)
        self._asset_urls: dict[str, str] = {}
        self.md = self._markdown()
        self.env = self._templates()

    # ------------------------------------------------------------------ URLs
    def url(self, path: str = "") -> str:
        """Root-relative URL for a site path ('' is the home page)."""
        return self.base + path.lstrip("/")

    def absolute(self, path: str = "") -> str:
        return self.origin + self.url(path)

    def github(self, repo_path: str, kind: str = "blob") -> str:
        cfg = self.config
        return f"https://github.com/{cfg['repo']}/{kind}/{cfg['branch']}/{repo_path}".rstrip("/")

    def asset(self, path: str) -> str:
        """URL of a file in site/static/, with a content hash so browsers never serve a stale copy."""
        if path not in self._asset_urls:
            digest = hashlib.sha1((SITE / "static" / path).read_bytes()).hexdigest()[:8]
            self._asset_urls[path] = f"{self.url('assets/' + path)}?v={digest}"
        return self._asset_urls[path]

    # ------------------------------------------------------------ discovery
    def is_excluded(self, path: Path) -> bool:
        rel = rel_path(path)
        return any(fnmatch.fnmatch(path.name, pat) or fnmatch.fnmatch(rel, pat) for pat in self.exclude)

    def walk(self, folder: Path):
        """Yield publishable files, skipping hidden folders and Python packages."""
        for current, dirs, files in os.walk(folder):
            here = Path(current)
            dirs[:] = sorted(
                d for d in dirs
                if not d.startswith((".", "__"))
                and not (here / d / "__init__.py").exists()
                and not self.is_excluded(here / d)
            )
            for name in sorted(files):
                path = here / name
                if not name.startswith(".") and not self.is_excluded(path):
                    yield path

    def discover(self) -> None:
        for folder in sorted((ROOT / "posts").iterdir()):
            if not folder.is_dir() or folder.name.startswith(("_", ".")):
                continue
            index = folder / "index.md"
            if not index.exists():
                raise BuildError(f"{rel_path(folder)}/ has no index.md")
            post = self.load(index, "post")
            if post.meta.get("draft") and not self.include_drafts:
                continue
            self.register(post)
            self.posts.append(post)
            for path in self.walk(folder):
                if path == index:
                    continue
                if path.suffix == ".md":
                    doc = self.load(path, "doc")
                    doc.post = post
                    post.docs.append(doc)
                    self.register(doc)
                else:
                    self.add_asset(path, rel_path(path))
            post.docs.sort(key=lambda d: d.title.lower())
        for path in self.walk(ROOT / "pages"):
            if path.suffix == ".md":
                self.register(self.load(path, "page"))
            else:
                self.add_asset(path, rel_path(path)[len("pages/"):])

        self.posts.sort(key=lambda p: p.date, reverse=True)
        for newer, older in zip(self.posts, self.posts[1:]):
            newer.prev, older.next = older, newer

    def register(self, page: Page) -> None:
        self.pages.append(page)
        self.url_of[page.rel] = page.url

    def add_asset(self, path: Path, site_path: str) -> None:
        self.assets[rel_path(path)] = site_path
        self.url_of[rel_path(path)] = site_path

    def load(self, path: Path, kind: str) -> Page:
        rel = rel_path(path)
        meta, body = split_front_matter(path.read_text(encoding="utf-8"))
        if kind == "post":
            url = rel[: -len("index.md")]
        elif kind == "page":
            name = rel[len("pages/"):-3]
            url = "" if name == "index" else f"{name}/"
        else:
            url = rel[:-3] + ".html"
        page = Page(source=path, kind=kind, url=url, meta=meta, body=body)

        if kind == "post":
            missing = [key for key in REQUIRED_POST_FIELDS if not meta.get(key)]
            if missing:
                raise BuildError(f"{rel}: front matter is missing {', '.join(missing)}")
            meta["date"] = as_date(meta["date"], path, "date")
            if meta.get("updated"):
                meta["updated"] = as_date(meta["updated"], path, "updated")
        if not meta.get("title"):
            # Appendices and pages may take their title from the first heading.
            heading = re.search(r"^#\s+(.+?)\s*#*\s*$", body, re.M)
            if not heading:
                raise BuildError(f"{rel}: needs a `title` in front matter or a `# Heading`")
            meta["title"] = heading.group(1).strip()
            body = body[: heading.start()] + body[heading.end():]
        page.body = body
        page.title = meta["title"]
        page.description = meta.get("description") or self.first_paragraph(body)
        return page

    @staticmethod
    def first_paragraph(markdown: str) -> str:
        for block in re.split(r"\n\s*\n", markdown):
            block = block.strip()
            if block and not block.startswith(("#", ">", "|", "```", "- ", "* ", "<", "!")):
                text = re.sub(r"\*\*|`|\[([^\]]*)\]\([^)]*\)", r"\1", " ".join(block.split()))
                return text if len(text) <= 200 else text[:197].rsplit(" ", 1)[0] + "…"
        return ""

    # ------------------------------------------------------------- markdown
    def _markdown(self) -> MarkdownIt:
        md = MarkdownIt("commonmark", {"html": True})
        md.enable(["table", "strikethrough"])
        md.options["alerts"] = True  # GitHub-style > [!NOTE] callouts
        md.options["tasklists"] = True
        md.use(footnote_plugin)
        md.use(anchors_plugin, min_level=2, max_level=6, slug_func=heading_slug,
               permalink=True, permalinkSymbol="#")
        formatter = HtmlFormatter(nowrap=True)

        def fence(renderer, tokens, idx, options, env):
            token = tokens[idx]
            info = unescapeAll(token.info).strip() if token.info else ""
            lang = info.split(maxsplit=1)[0].lower() if info else ""
            lexer = None
            if lang not in PLAIN_LANGS:
                try:
                    lexer = get_lexer_by_name(lang)
                except ClassNotFound:
                    pass
            code = highlight(token.content, lexer, formatter) if lexer else html.escape(token.content)
            label = CODE_LANG_LABELS.get(lang, lang)
            head = f'<div class="code-head"><span>{html.escape(label)}</span></div>' if lexer else ""
            return (f'<div class="code-block">{head}<pre><code class="language-{html.escape(lang or "text")}">'
                    f"{code}</code></pre></div>\n")

        def table_open(renderer, tokens, idx, options, env):
            return '<div class="table-wrap">\n' + renderer.renderToken(tokens, idx, options, env)

        def table_close(renderer, tokens, idx, options, env):
            return renderer.renderToken(tokens, idx, options, env) + "</div>\n"

        md.add_render_rule("fence", fence)
        md.add_render_rule("table_open", table_open)
        md.add_render_rule("table_close", table_close)
        return md

    def render(self, page: Page) -> None:
        env: dict = {}
        tokens = self.md.parse(page.body, env)
        self.rewrite_links(tokens, page)
        self.make_figures(tokens)
        page.toc = self.table_of_contents(tokens)
        page.content = Markup(self.md.renderer.render(tokens, self.md.options, env))

    def render_inline(self, text: str, page: Page) -> Markup:
        tokens = self.md.parseInline(str(text), {})
        self.rewrite_links(tokens, page)
        return Markup(self.md.renderer.render(tokens, self.md.options, {}))

    def rewrite_links(self, tokens: list[Token], page: Page) -> None:
        for token in tokens:
            for child in token.children or []:
                if child.type == "link_open":
                    if "header-anchor" in (child.attrGet("class") or ""):
                        child.attrSet("aria-label", "Link to this section")
                    else:
                        child.attrSet("href", self.resolve(str(child.attrGet("href") or ""), page))
                elif child.type == "image":
                    child.attrSet("src", self.resolve(str(child.attrGet("src") or ""), page))
                    child.attrSet("loading", "lazy")
                    child.attrSet("decoding", "async")

    def resolve(self, href: str, page: Page) -> str:
        """Map a link written against the repository layout to its published URL."""
        parts = urlsplit(href)
        if parts.scheme or parts.netloc or not parts.path or href.startswith("/"):
            return href
        suffix = ("?" + parts.query if parts.query else "") + ("#" + parts.fragment if parts.fragment else "")
        target = posixpath.normpath(posixpath.join(page.folder, unquote(parts.path)))
        if target.startswith(".."):
            return href
        if target == ".":
            return self.url("") + suffix
        if target in self.url_of:
            return self.url(self.url_of[target]) + suffix
        if f"{target}/index.md" in self.url_of:
            return self.url(self.url_of[f"{target}/index.md"]) + suffix
        path = ROOT / target
        if path.exists():  # part of the repository but not of the site: link to GitHub
            return self.github(target, "tree" if path.is_dir() else "blob") + suffix
        return href  # broken; reported by check_links()

    def make_figures(self, tokens: list[Token]) -> None:
        """A paragraph holding only an image becomes a <figure>; its title becomes the caption."""
        for i in range(len(tokens) - 2):
            opening, inline, closing = tokens[i:i + 3]
            if (opening.type, inline.type, closing.type) != ("paragraph_open", "inline", "paragraph_close"):
                continue
            children = [c for c in inline.children or [] if c.type != "softbreak" and (c.type != "text" or c.content.strip())]
            if len(children) != 1 or children[0].type != "image":
                continue
            image = children[0]
            caption = image.attrGet("title")
            if caption:
                del image.attrs["title"]
            link_open = Token("link_open", "a", 1)
            link_open.attrSet("href", str(image.attrGet("src")))
            link_open.attrSet("class", "figure-link")
            link_open.attrSet("title", "Open full-size image")
            inline.children = [link_open, image, Token("link_close", "a", -1)]
            if caption:
                rendered = self.md.renderInline(str(caption))
                inline.children.append(Token("html_inline", "", 0, content=f"<figcaption>{rendered}</figcaption>"))
            opening.tag = closing.tag = "figure"

    @staticmethod
    def table_of_contents(tokens: list[Token]) -> list[dict]:
        toc = []
        for i, token in enumerate(tokens):
            if token.type == "heading_open" and token.tag in ("h2", "h3"):
                text = "".join(c.content for c in tokens[i + 1].children or [] if c.type in ("text", "code_inline"))
                toc.append({"id": token.attrGet("id"), "text": text.strip(), "level": int(token.tag[1])})
        return toc

    # ------------------------------------------------------------ templates
    def _templates(self) -> Environment:
        env = Environment(
            loader=FileSystemLoader(SITE / "templates"),
            autoescape=select_autoescape(["html", "xml"]),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        env.globals.update(site=self.config, url=self.url, absolute=self.absolute, asset=self.asset, github=self.github)
        env.filters["human_date"] = human_date
        env.filters["slugify"] = slugify
        env.filters["iso"] = lambda d: d.isoformat()
        return env

    def write(self, site_path: str, text: str) -> None:
        target = self.out / (site_path + "index.html" if site_path == "" or site_path.endswith("/") else site_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def render_template(self, name: str, **context) -> str:
        return self.env.get_template(name).render(**context)

    def topics(self) -> list[dict]:
        topics: dict[str, dict] = {}
        for post in self.posts:
            for tag in post.meta["tags"]:
                entry = topics.setdefault(slugify(tag), {"name": tag, "slug": slugify(tag), "posts": []})
                entry["posts"].append(post)
        for entry in topics.values():
            entry["url"] = self.url(f"topics/{entry['slug']}/")
        return sorted(topics.values(), key=lambda t: (-len(t["posts"]), t["name"].lower()))

    # ---------------------------------------------------------------- build
    def build(self) -> list[str]:
        self.discover()
        for page in self.pages:
            self.render(page)

        staging = self.out.with_name(self.out.name + ".tmp")
        shutil.rmtree(staging, ignore_errors=True)
        final_out, self.out = self.out, staging
        try:
            self.write_site()
        finally:
            self.out = final_out
        shutil.rmtree(self.out, ignore_errors=True)
        staging.rename(self.out)
        return self.check_links()

    def write_site(self) -> None:
        topics = self.topics()
        topic_url = {tag: self.url(f"topics/{slugify(tag)}/") for post in self.posts for tag in post.meta["tags"]}
        common = {"topics": topics, "topic_url": topic_url, "posts": self.posts}

        for page in self.pages:
            if page.kind == "post":
                meta = page.meta
                html_out = self.render_template(
                    "post.html", page=page, **common,
                    reading_time=meta.get("reading_time") or reading_minutes(page.body),
                    key_findings=[self.render_inline(item, page) for item in meta.get("key_findings", [])],
                    resources=[dict(item, href=self.resolve(item["url"], page)) for item in self.resources(page)],
                )
            else:
                html_out = self.render_template("doc.html", page=page, **common)
            self.write(page.url, html_out)

        reading = {p.url: p.meta.get("reading_time") or reading_minutes(p.body) for p in self.posts}
        self.write("", self.render_template("home.html", page=None, reading=reading, **common))
        self.write("topics/", self.render_template("topics.html", page=None, reading=reading, topic=None, **common))
        for topic in topics:
            self.write(f"topics/{topic['slug']}/",
                       self.render_template("topics.html", page=None, reading=reading, topic=topic, **common))
        self.write("404.html", self.render_template("404.html", page=None, **common))
        self.write("feed.xml", self.render_template(
            "feed.xml", entries=[(post, self.absolutize(post.content)) for post in self.posts],
            feed_updated=max((p.updated or p.date for p in self.posts), default=dt.date.today()), **common))
        self.write("sitemap.xml", self.render_template("sitemap.xml", pages=self.pages, **common))
        self.write("robots.txt", f"User-agent: *\nAllow: /\nSitemap: {self.absolute('sitemap.xml')}\n")
        self.write(".nojekyll", "")

        for source, site_path in self.assets.items():
            target = self.out / site_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / source, target)
        shutil.copytree(SITE / "static", self.out / "assets", dirs_exist_ok=True)

    @staticmethod
    def resources(page: Page) -> list[dict]:
        items = page.meta.get("resources") or []
        for item in items:
            if not isinstance(item, dict) or not item.get("title") or not item.get("url"):
                raise BuildError(f"{page.rel}: every entry in `resources` needs a `title` and a `url`")
        return items

    def absolutize(self, content: str) -> str:
        """Feed readers need absolute URLs. Returns plain text so the feed template escapes it."""
        return re.sub(r'(href|src)="(/[^"]*)"', lambda m: f'{m.group(1)}="{self.origin}{m.group(2)}"', str(content))

    # ---------------------------------------------------------- link check
    def check_links(self) -> list[str]:
        errors: list[str] = []
        ids: dict[Path, set[str]] = {}

        def ids_in(path: Path) -> set[str]:
            if path not in ids:
                ids[path] = set(re.findall(r'\sid="([^"]+)"', path.read_text(encoding="utf-8")))
            return ids[path]

        for page_file in sorted(self.out.rglob("*.html")):
            page_path = page_file.relative_to(self.out).as_posix()
            text = page_file.read_text(encoding="utf-8")
            for raw in re.findall(r'\s(?:href|src)="([^"]*)"', text):
                href = html.unescape(raw)
                parts = urlsplit(href)
                if parts.scheme or parts.netloc or href.startswith(("mailto:", "data:")):
                    continue
                if not parts.path:  # same-page anchor
                    target = page_file
                elif href.startswith("/"):
                    if not parts.path.startswith(self.base):
                        errors.append(f"{page_path}: {href} is outside the site base path {self.base}")
                        continue
                    target = self.out / unquote(parts.path[len(self.base):])
                else:
                    target = page_file.parent / unquote(parts.path)
                if target.is_dir() or parts.path.endswith("/"):
                    target = target / "index.html"
                if not target.exists():
                    errors.append(f"{page_path}: broken link {href}")
                elif parts.fragment and target.suffix == ".html" and unquote(parts.fragment) not in ids_in(target):
                    errors.append(f"{page_path}: missing anchor #{parts.fragment} in {href or page_path}")
        return sorted(set(errors))


# --------------------------------------------------------------------- CLI
def build_once(out: Path, include_drafts: bool) -> list[str]:
    started = time.perf_counter()
    site = Site(out, include_drafts)
    errors = site.build()
    print(f"Built {len(site.pages)} pages ({len(site.posts)} posts) and {len(site.assets)} files "
          f"into {rel_path(out) if out.is_relative_to(ROOT) else out} in {time.perf_counter() - started:.2f}s")
    for error in errors:
        print(f"  ✗ {error}", file=sys.stderr)
    return errors


def source_state() -> tuple:
    state = []
    for folder in (ROOT / "posts", ROOT / "pages", SITE):
        for path in folder.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                state.append((str(path), path.stat().st_mtime_ns))
    return tuple(sorted(state))


def serve(out: Path, port: int, include_drafts: bool) -> None:
    import threading

    base = Site(out).base

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(out), **kwargs)

        def translate_path(self, path):
            path = urlsplit(path).path
            if path.startswith(base):
                path = "/" + path[len(base):]
            return super().translate_path(path)

        def do_GET(self):
            if base != "/" and not self.path.startswith(base):
                self.send_response(302)
                self.send_header("Location", base)
                self.end_headers()
                return
            super().do_GET()

        def send_error(self, code, message=None, explain=None):
            not_found = out / "404.html"
            if code == 404 and not_found.exists():
                body = not_found.read_bytes()
                self.send_response(404)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                super().send_error(code, message, explain)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Serving on http://localhost:{port}{base}  (Ctrl+C to stop)")
    state = source_state()
    try:
        while True:
            time.sleep(1)
            current = source_state()
            if current != state:
                state = current
                try:
                    build_once(out, include_drafts)
                except Exception as exc:  # keep serving; show the error and wait for the next edit
                    print(f"Build failed: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        server.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "_site", help="output folder (default: _site)")
    parser.add_argument("--serve", action="store_true", help="serve the site and rebuild on changes")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--drafts", action="store_true", help="include posts marked draft: true")
    args = parser.parse_args()
    out = args.out.resolve()
    try:
        errors = build_once(out, args.drafts)
    except BuildError as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        return 1
    if args.serve:
        serve(out, args.port, args.drafts)
        return 0
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
