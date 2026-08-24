#!/usr/bin/env python3
r"""Assemble index.html from template.html and the fragments in sections/.

Works like \input in LaTeX: a line in template.html of the form

    <!-- @include sections/bio.html -->

is replaced by the contents of that file, indented to match the include line.
Includes may nest. Run after editing anything under sections/:

    python3 build.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "template.html"
OUTPUT = ROOT / "index.html"

INCLUDE_RE = re.compile(r"^([ \t]*)<!--\s*@include\s+(\S+)\s*-->[ \t]*$")

BANNER = """<!--
================================================================================
GENERATED FILE - DO NOT EDIT DIRECTLY.
Edit template.html or the fragments in sections/, then run: python3 build.py
================================================================================
-->
"""


def expand(path, stack):
    """Return the text of `path` with every @include line expanded."""
    if path in stack:
        chain = " -> ".join(str(p.relative_to(ROOT)) for p in stack + [path])
        sys.exit("build.py: circular include: %s" % chain)

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        sys.exit("build.py: missing file: %s" % path.relative_to(ROOT))

    out = []
    for lineno, line in enumerate(text.splitlines(), 1):
        match = INCLUDE_RE.match(line)
        if not match:
            out.append(line)
            continue

        indent, target = match.group(1), match.group(2)
        included = expand((ROOT / target).resolve(), stack + [path])
        # Re-indent the fragment to sit where the @include line was.
        out.extend(indent + l if l.strip() else "" for l in included.splitlines())

    return "\n".join(out)


def main():
    body = expand(TEMPLATE, [])
    OUTPUT.write_text(BANNER + body.lstrip("\n").rstrip() + "\n", encoding="utf-8")
    print("build.py: wrote %s" % OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
