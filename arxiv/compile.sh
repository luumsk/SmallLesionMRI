#!/usr/bin/env bash
# Compile an arXiv paper version and collect everything into submission/<version>/.
#
# Usage:  ./compile.sh [version]        (default: v02 -> main_v02.tex)
#
# Output: submission/<version>/
#   main_<version>.pdf                  compiled paper
#   compile.log                         full pdflatex/bibtex console output
#   build/                              aux, log, bbl, blg, out, pdf
#   source/                             arXiv-ready sources (tex, sections, figures, bbl, bib)
#   arxiv_source_<version>.tar.gz       tarball of source/ for upload
set -euo pipefail

VERSION="${1:-v02}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
MAIN="main_${VERSION}"
OUT="$ROOT/submission/$VERSION"
BUILD="$OUT/build"
SRC="$OUT/source"
LOG="$OUT/compile.log"

for cmd in pdflatex bibtex; do
    command -v "$cmd" >/dev/null || { echo "error: $cmd not found" >&2; exit 1; }
done
[[ -f "$ROOT/$MAIN.tex" ]] || { echo "error: $ROOT/$MAIN.tex not found" >&2; exit 1; }

rm -rf "$OUT"
mkdir -p "$BUILD" "$SRC"
: > "$LOG"

run_pdflatex() {
    echo "===== pdflatex pass $1 =====" >> "$LOG"
    if ! (cd "$ROOT" && pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
            -output-directory "$BUILD" "$MAIN.tex") >> "$LOG" 2>&1; then
        echo "error: pdflatex pass $1 failed, see $LOG" >&2
        grep -E "^.*:[0-9]+: |^! " "$LOG" | tail -5 >&2 || true
        missing=$(grep -oE "File \`[^']+\.sty' not found" "$LOG" | sed -E "s/File \`(.*)\.sty' not found/\1/" | sort -u | tr '\n' ' ')
        [[ -n "$missing" ]] && echo "hint: tlmgr --usermode install $missing" >&2
        exit 1
    fi
}

echo "Compiling $MAIN.tex -> $OUT"
run_pdflatex 1
echo "===== bibtex =====" >> "$LOG"
(cd "$BUILD" && BIBINPUTS="$ROOT:" bibtex "$MAIN") >> "$LOG" 2>&1 \
    || { echo "error: bibtex failed, see $LOG" >&2; exit 1; }
run_pdflatex 2
run_pdflatex 3
# Rerun until cross-references are stable (at most 3 extra passes)
for pass in 4 5 6; do
    grep -q "Rerun to get" "$BUILD/$MAIN.log" || break
    run_pdflatex "$pass"
done

cp "$BUILD/$MAIN.pdf" "$OUT/"

# ---------- arXiv source bundle ----------
cp "$ROOT/$MAIN.tex" "$SRC/"
cp "$BUILD/$MAIN.bbl" "$SRC/"   # arXiv does not run bibtex; it needs the .bbl
# Section directories and bib file referenced by the main file
for dir in $(grep -oE '\\input\{[^}/]+/' "$ROOT/$MAIN.tex" | sed -E 's/\\input\{//; s#/$##' | sort -u); do
    cp -R "$ROOT/$dir" "$SRC/"
done
for bib in $(grep -oE '\\bibliography\{[^}]+\}' "$ROOT/$MAIN.tex" | sed -E 's/\\bibliography\{//; s/\}$//' | tr ',' ' '); do
    cp "$ROOT/${bib%.bib}.bib" "$SRC/"
done
cp -R "$ROOT/figures" "$SRC/"
find "$SRC" \( -name .DS_Store -o -name '._*' \) -delete
# Keep macOS metadata out of the bundle (otherwise bsdtar adds ._* AppleDouble entries)
COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs -czf "$OUT/arxiv_source_${VERSION}.tar.gz" -C "$SRC" .

# ---------- summary ----------
undefined=$(grep -cE "undefined" "$BUILD/$MAIN.log" || true)
overfull=$(grep -cE "^Overfull" "$BUILD/$MAIN.log" || true)
pages=$(tr -d '\n' < "$BUILD/$MAIN.log" | grep -oE "Output written on [^(]*\([0-9]+ pages" | grep -oE "[0-9]+ pages" || true)
echo "Done: $OUT/$MAIN.pdf (${pages:-? pages})"
echo "  undefined refs/citations: $undefined, overfull boxes: $overfull"
echo "  full log: $LOG"
echo "  arXiv bundle: $OUT/arxiv_source_${VERSION}.tar.gz"
