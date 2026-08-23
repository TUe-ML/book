#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
import hashlib
import shutil
import subprocess
import tempfile

from pathlib import Path

plugin = {
    "name": "Tikz to SVG",
    "directives": [
        {
            "name": "tikz",
            "doc": "A directive to compile tikz into svg.",
            "alias": ["tikz2svg"],
            "body": {
                "type": "string",
                "doc": "The tikz source code.",
            },
        }
    ],
}

DOC_HEAD = r"""
\documentclass[crop,12pt,tikz]{standalone}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{pgfplots}
\usetikzlibrary{%s}
\usepackage{pgfplotstable}
\pagestyle{empty}
"""

DOC_BODY = r"""
\begin{document}
%s
\end{document}
"""


def generate_latex(tikz_source, libraries="", preamble=""):
    """Generate a standalone LaTeX document containing TikZ source."""
    tikz = tikz_source.replace("\r\n", "\n")
    if "\\begin{tikzpicture}" not in tikz:
        tikz = "\\begin{tikzpicture}\n" + tikz + "\n\\end{tikzpicture}"

    return DOC_HEAD % libraries + preamble + DOC_BODY % tikz


def compile_latex(latex_source, workdir=None, latex_engine="pdflatex",
                  basename="tikz"):
    """Compile LaTeX source and return the generated PDF path."""
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="tikz-")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    
    tex_path = workdir / f"{basename}.tex"
    tex_path.write_text(latex_source, encoding="utf-8")
    subprocess.run(
        [latex_engine, "--interaction=nonstopmode", tex_path.name],
        cwd=workdir,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return workdir / f"{basename}.pdf"


def convert_pdf_to_svg(pdf_path, svg_path=None, pdf2svg="pdf2svg"):
    """Convert a PDF to SVG with pdf2svg and return the SVG path."""
    pdf_path = Path(pdf_path)
    if svg_path is None:
        svg_path = pdf_path.with_suffix(".svg")
    svg_path = Path(svg_path)
    subprocess.run(
        [pdf2svg, str(pdf_path), str(svg_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return svg_path


def move_svg_to_images(svg_path, images_dir="_svg_files", filename=None):
    """Move an SVG into the images directory and return its final path."""
    svg_path = Path(svg_path)
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    destination = images_dir / (filename or svg_path.name)
    if svg_path.resolve() != destination.resolve():
        shutil.move(str(svg_path), str(destination))
    return str(destination)

def tikz_to_svg(tikz_source, libraries="", preamble="", workdir=None, latex_engine="pdflatex", basename='tikz'):
    """Convert TikZ source to an SVG image and return the SVG path."""
    latex_source = generate_latex(tikz_source, libraries=libraries, preamble=preamble)
    hash = hashlib.sha256(latex_source.encode("utf-8")).hexdigest()
    basename += f"-{hash[:8]}"

    pdf_path = compile_latex(latex_source, workdir=workdir, latex_engine=latex_engine, basename=basename)
    svg_path = convert_pdf_to_svg(pdf_path)
    final_svg_path = move_svg_to_images(svg_path)
    return final_svg_path


# if __name__ == "__main__":
#     # Example usage
#     tikz_source = r"""
#     \draw[thick,->] (0,0) -- (1,2);
#     \draw[thick,->] (0,0) -- (1,-1);
#     """
#     final_svg_path = tikz_to_svg(tikz_source)
#     # latex_source = generate_latex(tikz_source)
#     # pdf_path = compile_latex(latex_source)
#     # svg_path = convert_pdf_to_svg(pdf_path)
#     # final_svg_path = move_svg_to_images(svg_path)
#     print(f"Generated SVG: {final_svg_path}")


def declare_result(content):
    """Declare result as JSON to stdout

    :param content: content to declare as the result
    """

    # Format result and write to stdout
    json.dump(content, sys.stdout, indent=2)
    # Successfully exit
    raise SystemExit(0)

def run_directive(name, data):
    """Execute a directive with the given name and data

    :param name: name of the directive to run
    :param data: data of the directive to run
    """
    assert name == "tikz"
    tikz_picture = data.get("body")
    svg_path = tikz_to_svg(
        tikz_picture,
        libraries = "calc, angles, intersections, arrows.meta, matrix, decorations.pathreplacing, positioning, shapes.geometric, fit, pgfplots.groupplots, pgfplots.fillbetween, 3d, decorations.text, shapes.arrows, backgrounds",
    # tikz_latex_preamble: \usepackage{pgfplotstable}
    )
    
    return [{
        "type": "image",
        "url": svg_path,
    }
    ]
    
def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--role")
    group.add_argument("--directive")
    group.add_argument("--transform")
    args = parser.parse_args()

    if args.directive:
        data = json.load(sys.stdin)
        declare_result(run_directive(args.directive, data))
    elif args.transform:
        raise NotImplementedError
    elif args.role:
        raise NotImplementedError
    else:
        declare_result(plugin)

if __name__ == "__main__":
    main()
