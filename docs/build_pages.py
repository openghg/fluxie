"""
Generate the parts of the documentation built from code and data.

- The gallery: runs the examples of `docs/gallery.py` and saves the figures.
- The publications of the hall of fame, read from `docs/hall_of_fame.bib`.

The results are written in `docs/generated/` and included in the pages
`docs/gallery.md` and `docs/hall_of_fame.md`. Run it before building the
documentation:

    python docs/build_pages.py
    zensical build
"""

import logging
import re
import shutil
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gallery

logger = logging.getLogger("build_pages")

DOCS_DIR = Path(__file__).parent
GENERATED_DIR = DOCS_DIR / "generated"
GALLERY_IMAGES_DIR = GENERATED_DIR / "gallery"
BIB_FILE = DOCS_DIR / "hall_of_fame.bib"


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _save_figure(example_func, path: Path) -> None:
    plt.close("all")
    result = example_func()
    fig = result[0] if isinstance(result, (tuple, list)) else result
    if fig is None:
        fig = plt.gcf()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close("all")


def build_gallery() -> None:
    shutil.rmtree(GALLERY_IMAGES_DIR, ignore_errors=True)
    GALLERY_IMAGES_DIR.mkdir(parents=True)

    lines = []
    current_section = None
    for section, title, plot_function, example_func in gallery.EXAMPLES:
        if section != current_section:
            lines.append(f"## {section}\n")
            current_section = section

        image = GALLERY_IMAGES_DIR / f"{_slug(section)}_{_slug(title)}.png"
        api_path = f"{plot_function.__module__}.{plot_function.__qualname__}"
        lines.append(f"### {title}\n")
        lines.append(f"[`{plot_function.__name__}`][{api_path}]\n")

        logger.info(f"Gallery: rendering '{title}'")
        try:
            _save_figure(example_func, image)
        except Exception as e:
            logger.warning(f"Gallery example '{title}' failed: {e!r}")
            lines.append("!!! failure\n    This example failed to run.\n")
        else:
            # Path relative to docs/gallery.md, which includes this file
            lines.append(f"![{title}]({image.relative_to(DOCS_DIR).as_posix()})\n")

    (GENERATED_DIR / "gallery.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Hall of fame
# ---------------------------------------------------------------------------

# LaTeX accent commands and the corresponding unicode combining characters
_ACCENTS = {
    '"': "\u0308",
    "'": "\u0301",
    "`": "\u0300",
    "^": "\u0302",
    "~": "\u0303",
    "=": "\u0304",
    ".": "\u0307",
    "c": "\u0327",
    "v": "\u030c",
    "u": "\u0306",
    "H": "\u030b",
}
_SPECIAL_CHARS = {
    r"\ss": "ß",
    r"\o": "ø",
    r"\O": "Ø",
    r"\aa": "å",
    r"\AA": "Å",
    r"\ae": "æ",
    r"\AE": "Æ",
    r"\l": "ł",
    r"\L": "Ł",
    r"\&": "&",
    r"\%": "%",
    r"\_": "_",
}

_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_SUPERSCRIPTS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _latex_to_text(text: str) -> str:
    """Convert the LaTeX commands commonly found in BibTeX to plain text."""

    def accent(match: re.Match) -> str:
        return unicodedata.normalize("NFC", match[2] + _ACCENTS[match[1]])

    text = re.sub(r"\\([\"'`^~=.]|[cvuH](?=[{ ]))\s*\{?\s*([A-Za-z])\}?", accent, text)
    for command, char in _SPECIAL_CHARS.items():
        # Letter commands (e.g. \o) must not be followed by another letter
        end = r"(?![A-Za-z])\s*(\{\})?" if command[-1].isalpha() else ""
        text = re.sub(re.escape(command) + end, char, text)
    # Math subscripts and superscripts, e.g. CH$_4$
    text = re.sub(r"\$_\{?(\d+)\}?\$", lambda m: m[1].translate(_SUBSCRIPTS), text)
    text = re.sub(r"\$\^\{?(\d+)\}?\$", lambda m: m[1].translate(_SUPERSCRIPTS), text)
    # Remove the remaining commands, e.g. formatting like {\it ...} or \emph{...}
    text = re.sub(r"\\[A-Za-z]+\s*", "", text)
    text = text.replace("---", "—").replace("--", "–")
    text = text.replace("{", "").replace("}", "").replace("$", "")
    return re.sub(r"\s+", " ", text).strip()


def _matching_brace(text: str, start: int) -> int:
    """Index after the brace closing the one opened just before `start`."""
    depth = 1
    i = start
    while depth:
        if i >= len(text):
            raise ValueError("Unbalanced braces")
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return i


def _parse_fields(body: str) -> dict[str, str]:
    fields = {}
    i = 0
    field_name = re.compile(r"\s*,?\s*([A-Za-z][\w-]*)\s*=\s*")
    while match := field_name.match(body, i):
        name = match[1].lower()
        i = match.end()
        if body[i] == "{":
            end = _matching_brace(body, i + 1)
            value = body[i + 1 : end - 1]
        elif body[i] == '"':
            # Quotes inside braces, e.g. {\"u}, do not close the value
            end = (
                re.compile(r'"((?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|[^"{}])*)"')
                .match(body, i)
                .end()
            )
            value = body[i + 1 : end - 1]
        else:
            end = re.compile(r"[^,}]*").match(body, i).end()
            value = body[i:end]
        fields[name] = value.strip()
        i = end
    return fields


def parse_bibtex(text: str) -> list[dict[str, str]]:
    """
    Parse the entries of a BibTeX file.

    Returns:
        entries:
            One dictionary per entry, with the lowercase field names as keys,
            and the raw text of the entry under "bibtex".
    """
    entries = []
    for match in re.finditer(r"^\s*@(\w+)\s*\{", text, re.MULTILINE):
        if match[1].lower() in ("comment", "string", "preamble"):
            continue
        try:
            end = _matching_brace(text, match.end())
        except ValueError:
            line = text[match.start() :].strip().splitlines()[0]
            raise ValueError(
                f"Unbalanced braces in the BibTeX entry starting with: {line}"
            ) from None
        _, _, body = text[match.end() : end - 1].partition(",")
        entry = _parse_fields(body)
        entry["bibtex"] = text[match.start() : end].strip()
        entries.append(entry)
    return entries


def _short_authors(authors: str) -> str:
    names = [
        name.split(",")[0] if "," in name else name.split()[-1]
        for name in re.split(r"\s+and\s+", _latex_to_text(authors))
    ]
    if len(names) > 2:
        return f"{names[0]} et al."
    return " and ".join(names)


def _format_publication(entry: dict[str, str]) -> str:
    title = _latex_to_text(entry.get("title", "Untitled"))
    citation = _short_authors(entry.get("author", "Unknown"))
    citation += f" ({entry.get('year', 'n.d.')})."
    venue = next(
        (entry[k] for k in ("journal", "booktitle", "publisher") if k in entry), None
    )
    if venue:
        details = [f"*{_latex_to_text(venue)}*"]
        details += [_latex_to_text(entry[k]) for k in ("volume", "pages") if k in entry]
        citation += " " + ", ".join(details) + "."
    if "doi" in entry:
        citation += f" [doi:{entry['doi']}](https://doi.org/{entry['doi']})"
    elif "url" in entry:
        citation += f" [{entry['url']}]({entry['url']})"

    return f"""### {title}

{citation}

<details markdown>
<summary>BibTeX</summary>

```bibtex
{entry['bibtex']}
```

</details>
"""


def build_hall_of_fame() -> None:
    entries = parse_bibtex(BIB_FILE.read_text(encoding="utf-8"))
    # Most recent first, keeping the order of the file within a year
    entries.sort(key=lambda e: -int(e.get("year", "0")[:4] or 0))
    logger.info(f"Hall of fame: {len(entries)} publications")
    content = "\n".join(_format_publication(entry) for entry in entries)
    (GENERATED_DIR / "publications.md").write_text(content, encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    GENERATED_DIR.mkdir(exist_ok=True)
    build_hall_of_fame()
    build_gallery()
