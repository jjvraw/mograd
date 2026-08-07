#!/usr/bin/env python3
"""Convert `mojo doc` JSON output into Docusaurus markdown pages.

One page per module. Structs, traits and free functions become h2 sections, a
type's Parameters/Fields/Aliases/Methods become h3, and individual methods h4.
Pages are emitted as CommonMark `.md` so no MDX escaping is needed (the site
sets `markdown.format: 'detect'`).

    mojo doc -I . -o mograd.json mograd/
    python3 scripts/gen_api_docs.py mograd.json -o docs/docs/api
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

CODE_FENCE = "mojo"

# ===-------------------------------------------------------------------===#
# Site layout
#
# Which subtrees are implementation detail, and the order public entries
# appear in the sidebar. Anything not listed in ORDER sorts alphabetically
# after everything that is.
# ===-------------------------------------------------------------------===#

INTERNAL = [
    "mograd.op",
    "mograd.grad",
    "mograd.pattern_matcher",
    "mograd.scheduler",
    "mograd.simplify",
    "mograd.runtime",
]

LABEL = "API Reference"

ORDER = [
    "mograd.tensor",
    "mograd.nn",
    "mograd.device_ctx",
    "mograd.data",
    "mograd.debug",
    "mograd.buffer",
    "mograd.testing",
    "mograd.layout",
]


# ===-------------------------------------------------------------------===#
# Text helpers
# ===-------------------------------------------------------------------===#


def flatten(text: str) -> str:
    """Collapse a docstring fragment onto a single line."""
    return " ".join((text or "").split())


def code(text: str) -> str:
    """Wrap a value in backticks, widening the fence if it contains one."""
    text = (text or "").strip()
    if not text:
        return ""
    fence = "``" if "`" in text else "`"
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def slug(*parts: str) -> str:
    """Build a stable heading anchor from name parts."""
    joined = "-".join(p for p in parts if p)
    out = []
    for ch in joined.lower():
        if ch.isalnum() or ch == "-":
            out.append(ch)
        elif ch == "_":
            out.append("-")
    return "-".join(filter(None, "".join(out).split("-")))


def block(doc: dict[str, Any]) -> list[str]:
    """Render the summary and description of a declaration."""
    lines = []
    summary = (doc.get("summary") or "").strip()
    description = (doc.get("description") or "").strip()
    if summary:
        lines += [summary, ""]
    if description and description != summary:
        lines += [description, ""]
    return lines


def entry(decl: dict[str, Any]) -> str:
    """Render one parameter, arg, field or alias as `name (Type): description`.

    Each key is optional, which is what lets the four kinds share this: aliases
    carry `value` and no `type`, the rest carry `type` and no `value`. Defaults
    are skipped because the signature above already shows them.
    """
    line = f"- {code(decl.get('name', ''))}"
    if decl.get("type"):
        line += f" ({code(decl['type'])})"
    if decl.get("value"):
        line += f" = {code(decl['value'])}"
    description = flatten(decl.get("summary") or decl.get("description", ""))
    if description:
        line += f": {description}"
    return line


def items(decls: list[dict[str, Any]]) -> list[str]:
    """Render declarations as definition items, dropping the implicit `self`."""
    return [entry(d) for d in decls if d.get("name") != "self"]


def definitions(label: str, items: list[str]) -> list[str]:
    """Wrap definition-list items under a heading."""
    if not items:
        return []
    return [label, ""] + items + [""]


# ===-------------------------------------------------------------------===#
# Declaration rendering
# ===-------------------------------------------------------------------===#


def heading(level: int, title: str, anchor: str) -> str:
    return f"{'#' * level} {title} {{#{anchor}}}"


def render_overload(ov: dict[str, Any]) -> list[str]:
    """Render one function overload: signature, docs, args and returns.

    Sections inside a function are bold labels rather than headings, so they
    stay out of the page table of contents.
    """
    lines = []
    signature = (ov.get("signature") or ov.get("name", "")).strip()
    lines += [f"```{CODE_FENCE}", signature, "```", ""]

    deprecated = (ov.get("deprecated") or "").strip()
    if deprecated:
        lines += [f":::warning[Deprecated]\n{deprecated}\n:::", ""]

    lines += block(ov)
    lines += definitions("**Parameters:**", items(ov.get("parameters", [])))
    lines += definitions("**Args:**", items(ov.get("args", [])))

    returns = ov.get("returns") or {}
    ret_doc = flatten(returns.get("doc", ""))
    ret_type = (returns.get("type") or "").strip()
    if ret_type == "Self" and ov.get("name") in ("__init__", "__copyinit__", "__moveinit__"):
        ret_type = ""
    if ret_type and ret_type != "None":
        rendered = code(ret_type)
        if ret_doc:
            rendered += f": {ret_doc}"
        lines += ["**Returns:**", "", rendered, ""]

    raises_doc = flatten(ov.get("raisesDoc", ""))
    if raises_doc:
        lines += ["**Raises:**", "", raises_doc, ""]

    constraints = flatten(ov.get("constraints", ""))
    if constraints:
        lines += ["**Constraints:**", "", constraints, ""]

    return lines


def render_function(fn: dict[str, Any], level: int, *parents: str) -> list[str]:
    """Render a function and all of its overloads under a single heading."""
    name = fn.get("name", "")
    anchor = slug(*parents, name)
    lines = [f"{'#' * level} {code(name)} {{#{anchor}}}", ""]
    overloads = fn.get("overloads", [])
    for i, ov in enumerate(overloads):
        if len(overloads) > 1:
            lines += [f"*Overload {i + 1} of {len(overloads)}*", ""]
        lines += render_overload(ov)
    return lines


def render_type(decl: dict[str, Any], level: int) -> list[str]:
    """Render a struct or trait with its fields, aliases and methods.

    Sections on a type are real headings so they reach the table of contents,
    which puts methods one level deeper than the type itself.
    """
    name = decl.get("name", "")
    anchor = slug(name)
    lines = [f"{'#' * level} {code(name)} {{#{anchor}}}", ""]

    signature = (decl.get("signature") or "").strip()
    if signature and signature != name:
        lines += [f"```{CODE_FENCE}", signature, "```", ""]

    deprecated = (decl.get("deprecated") or "").strip()
    if deprecated:
        lines += [f":::warning[Deprecated]\n{deprecated}\n:::", ""]

    lines += block(decl)

    constraints = flatten(decl.get("constraints", ""))
    if constraints:
        lines += ["**Constraints:**", "", constraints, ""]

    sub = level + 1

    params = items(decl.get("parameters", []))
    if params:
        lines += definitions(heading(sub, "Parameters", f"{anchor}-parameters"), params)

    fields = items(decl.get("fields", []))
    if fields:
        lines += definitions(heading(sub, "Fields", f"{anchor}-fields"), fields)

    aliases = items(decl.get("aliases", []))
    if aliases:
        lines += definitions(heading(sub, "Aliases", f"{anchor}-aliases"), aliases)

    traits = [code(p.get("name", "")) for p in decl.get("parentTraits", []) if p.get("name")]
    if traits:
        lines += [heading(sub, "Implemented traits", f"{anchor}-traits"), "", ", ".join(traits), ""]

    functions = decl.get("functions", [])
    if functions:
        lines += [heading(sub, "Methods", f"{anchor}-methods"), ""]
        for fn in functions:
            lines += render_function(fn, sub + 1, name)

    return lines


# ===-------------------------------------------------------------------===#
# Page assembly
# ===-------------------------------------------------------------------===#


def frontmatter(title: str, label: str, description: str, position: int | None = None) -> list[str]:
    lines = ["---", f"title: {json.dumps(title)}", f"sidebar_label: {json.dumps(label)}"]
    if description:
        lines.append(f"description: {json.dumps(description)}")
    if position is not None:
        lines.append(f"sidebar_position: {position}")
    lines += ["---", ""]
    return lines


def module_is_empty(mod: dict[str, Any]) -> bool:
    return not any(mod.get(k) for k in ("structs", "traits", "functions", "aliases"))


def render_declarations(node: dict[str, Any]) -> list[str]:
    """Render everything declared in a module or a package's `__init__`."""
    lines = []
    for struct in node.get("structs", []):
        lines += render_type(struct, 2)
    for trait in node.get("traits", []):
        lines += render_type(trait, 2)
    for fn in node.get("functions", []):
        lines += render_function(fn, 2)
    return lines + definitions(heading(2, "Aliases", "aliases"), items(node.get("aliases", [])))


def render_module(mod: dict[str, Any], qualified: str, position: int | None = None) -> str:
    """Render a whole module page."""
    summary = (mod.get("summary") or "").strip()
    lines = frontmatter(qualified, mod.get("name", ""), summary, position)

    lines += [f"# {qualified}", ""]
    lines += block(mod)
    lines += [f"```{CODE_FENCE}", f"from {qualified} import ...", "```", ""]
    lines += render_declarations(mod)

    return "\n".join(lines).rstrip() + "\n"


def render_index(
    title: str,
    children: list[tuple[str, str, str]],
    blurb: str = "",
    init: dict[str, Any] | None = None,
    position: int | None = None,
) -> str:
    """Render an index page listing members, for a package or a whole section.

    Declarations living in a package's own `__init__.mojo` are rendered inline
    here rather than getting a page of their own.
    """
    summary = (init.get("summary") or "").strip() if init else ""
    description = summary or blurb or f"API reference for {title}."
    lines = frontmatter(title, title.rsplit(".", 1)[-1], description, position)
    lines += [f"# {title}", ""]
    if init:
        lines += block(init)
    elif blurb:
        lines += [blurb, ""]

    items = []
    for name, link, desc in children:
        item = f"- [{name}]({link})"
        desc = flatten(desc)
        if desc:
            item += f"<br />\n  {desc}"
        items.append(item)
    lines += definitions("**Members**", items)

    if init:
        lines += render_declarations(init)
    return "\n".join(lines).rstrip() + "\n"


# ===-------------------------------------------------------------------===#
# Tree walk
# ===-------------------------------------------------------------------===#


class Generator:
    def __init__(self, exclude: list[str], order: list[str] | None = None) -> None:
        self.exclude = exclude
        self.order = order or []
        self.pages = 0

    def position_of(self, qualified: str) -> int | None:
        """Sidebar position for a qualified name, or None to sort alphabetically."""
        try:
            return self.order.index(qualified) + 1
        except ValueError:
            return None

    def excluded(self, qualified: str) -> bool:
        return any(qualified == e or qualified.startswith(e + ".") for e in self.exclude)

    def walk(self, pkg: dict[str, Any], qualified: str, directory: Path, position: int | None = None) -> None:
        if self.excluded(qualified):
            return

        directory.mkdir(parents=True, exist_ok=True)
        children: list[tuple[str, str, str]] = []

        for sub in sorted(pkg.get("packages", []), key=lambda p: p.get("name", "")):
            name = sub.get("name", "")
            if self.excluded(f"{qualified}.{name}"):
                continue
            self.walk(sub, f"{qualified}.{name}", directory / name)

            children.append((name, f"./{name}/", (sub.get("summary") or "").strip()))

        init = None
        for mod in sorted(pkg.get("modules", []), key=lambda m: m.get("name", "")):
            name = mod.get("name", "")
            if name == "__init__":
                if not module_is_empty(mod):
                    init = mod
                continue
            if self.excluded(f"{qualified}.{name}") or module_is_empty(mod):
                continue
            (directory / f"{name}.md").write_text(
                render_module(mod, f"{qualified}.{name}", self.position_of(f"{qualified}.{name}"))
            )
            self.pages += 1
            children.append((name, f"./{name}", (mod.get("summary") or "").strip()))

        label = qualified.rsplit(".", 1)[-1]
        category: dict[str, Any] = {"label": label, "collapsed": True}
        category_position = self.position_of(qualified)
        if category_position is not None:
            category["position"] = category_position
        (directory / "_category_.json").write_text(json.dumps(category, indent=2) + "\n")
        (directory / "index.md").write_text(render_index(qualified, children, init=init, position=position))
        self.pages += 1


def index_nodes(pkg: dict[str, Any], qualified: str, out: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map every qualified name in the tree to its declaration node."""
    out[qualified] = pkg
    for mod in pkg.get("modules", []):
        name = mod.get("name", "")
        if name != "__init__":
            out[f"{qualified}.{name}"] = mod
    for sub in pkg.get("packages", []):
        index_nodes(sub, f"{qualified}.{sub.get('name', '')}", out)
    return out


def write_category(directory: Path, label: str, position: int, collapsed: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "_category_.json").write_text(
        json.dumps({"label": label, "position": position, "collapsed": collapsed}, indent=2) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json", type=Path, help="mojodoc JSON produced by `mojo doc -o`")
    parser.add_argument("-o", "--out", type=Path, required=True, help="output directory for markdown pages")
    args = parser.parse_args()

    root = json.loads(args.json.read_text())["decl"]
    root_name = root.get("name", "api")

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    pages = 0
    sections: list[tuple[str, str, str]] = []

    # Public: the whole tree minus anything routed to Internals.
    public_dir = args.out / "public"
    public = Generator(INTERNAL, ORDER)
    public.walk(root, root_name, public_dir, position=0)
    write_category(public_dir, "Public", 1, False)
    pages += public.pages
    sections.append(("Public", "./public/", "The supported surface: tensors, autograd, layers and data."))

    # Internals: each requested subtree, rooted directly under internals/.
    if INTERNAL:
        nodes = index_nodes(root, root_name, {})
        internals_dir = args.out / "internals"
        internals_dir.mkdir(parents=True, exist_ok=True)
        internals = Generator([])
        children: list[tuple[str, str, str]] = []

        for qualified in INTERNAL:
            node = nodes.get(qualified)
            if node is None:
                print(f"gen_api_docs: warning: INTERNAL entry {qualified} matched nothing", file=sys.stderr)
                continue
            name = qualified.rsplit(".", 1)[-1]
            summary = flatten(node.get("summary", ""))
            if node.get("kind") == "package":
                internals.walk(node, qualified, internals_dir / name)
                children.append((name, f"./{name}/", summary))
            elif not module_is_empty(node):
                (internals_dir / f"{name}.md").write_text(render_module(node, qualified))
                internals.pages += 1
                children.append((name, f"./{name}", summary))

        write_category(internals_dir, "Internals", 2, True)
        (internals_dir / "index.md").write_text(
            render_index(
                "Internals",
                children,
                "Implementation detail. These modules back the public API and change without notice.",
                position=0,
            )
        )
        internals.pages += 1
        pages += internals.pages
        sections.append(("Internals", "./internals/", "Kernels, scheduling and storage that back the public API."))

    write_category(args.out, LABEL, 2, False)
    (args.out / "index.md").write_text(
        render_index(LABEL, sections, f"Generated from the {root_name} source docstrings.", position=0)
    )
    pages += 1

    print(f"gen_api_docs: wrote {pages} pages to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
