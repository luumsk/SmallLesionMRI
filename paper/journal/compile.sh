#!/usr/bin/env bash
# Compile the MDPI journal version of the paper and collect a submission bundle.
#
# Usage:  ./compile.sh [label]          (default label: today's date, YYYY-MM-DD)
#
# Environment overrides:
#   JOURNAL=<mdpi journal option>       default: the option in main.tex (biomedicines)
#   TEMPLATE=<path to MDPI template>    default: ../MDPI_template_ACS
#   FIGURES=<path to figure folder>     default: ./figures if it exists, else ../arxiv/figures
#
# Output: submission/<label>/
#   main.pdf                            compiled paper
#   compile.log                         full pdflatex/bibtex console output
#   source/                             self-contained LaTeX sources (compiled here):
#                                       main.tex, sections/, figures/, Definitions/, references.bib, main.bbl
#   build/                              aux, log, bbl, blg, out
#   latex_source_<label>.zip            zip of source/ for upload to the MDPI system
#   supplementary_materials_<label>.zip CSV tables S1-S3 + README from ./supplementary (if present)
set -euo pipefail

LABEL="${1:-$(date +%F)}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="${TEMPLATE:-$ROOT/../MDPI_template_ACS}"
if [[ -z "${FIGURES:-}" ]]; then
    if [[ -d "$ROOT/figures" ]]; then FIGURES="$ROOT/figures"; else FIGURES="$ROOT/../arxiv/figures"; fi
fi
MAIN="main"
OUT="$ROOT/submission/$LABEL"
SRC="$OUT/source"
BUILD="$OUT/build"
LOG="$OUT/compile.log"

for cmd in pdflatex bibtex zip; do
    command -v "$cmd" >/dev/null || { echo "error: $cmd not found" >&2; exit 1; }
done
[[ -f "$ROOT/$MAIN.tex" ]] || { echo "error: $ROOT/$MAIN.tex not found" >&2; exit 1; }
[[ -f "$TEMPLATE/Definitions/mdpi.cls" ]] || { echo "error: MDPI class not found in $TEMPLATE/Definitions" >&2; exit 1; }
[[ -d "$FIGURES" ]] || { echo "error: figure folder $FIGURES not found" >&2; exit 1; }

rm -rf "$OUT"
mkdir -p "$SRC" "$BUILD"
: > "$LOG"

# ---------- stage a self-contained source tree ----------
cp "$ROOT/$MAIN.tex" "$SRC/"
cp -R "$ROOT/sections" "$SRC/sections"
# mdpi.bst abbreviates given names byte by byte, which breaks UTF-8 initials (e.g. "Ž."),
# so accented letters in the staged copy are rewritten as LaTeX escapes such as {\v{Z}}.
python3 - "$ROOT/references.bib" "$SRC/references.bib" <<'PY'
import sys, unicodedata
accents = {"́": "'", "̀": "`", "̂": "^", "̈": '"', "̌": "v",
           "̃": "~", "̧": "c", "̆": "u", "̊": "r", "̄": "="}
special = {"ß": r"{\ss}", "Ø": r"{\O}", "ø": r"{\o}", "Æ": r"{\AE}", "æ": r"{\ae}",
           "Å": r"{\AA}", "å": r"{\aa}", "Ł": r"{\L}", "ł": r"{\l}", "ı": r"{\i}"}
def esc(ch):
    if ch in special:
        return special[ch]
    d = unicodedata.normalize("NFD", ch)
    if len(d) == 2 and d[0].isascii() and d[0].isalpha() and d[1] in accents:
        base = r"\i" if d[0] == "i" else d[0]
        return "{\\%s{%s}}" % (accents[d[1]], base)
    return ch
text = open(sys.argv[1], encoding="utf-8").read()
open(sys.argv[2], "w", encoding="utf-8").write("".join(esc(c) for c in text))
PY
cp -R "$TEMPLATE/Definitions" "$SRC/Definitions"
mkdir -p "$SRC/figures"
# Copy only the figures that the paper uses (\includegraphics{name} with or without extension)
grep -rhoE '\\includegraphics(\[[^]]*\])?\{[^}]+\}' "$SRC/$MAIN.tex" "$SRC/sections" \
    | sed -E 's/.*\{([^}]+)\}$/\1/; s#^figures/##' | sort -u | while read -r fig; do
    found=0
    for f in "$FIGURES/$fig" "$FIGURES/$fig".{pdf,png,jpg,jpeg,eps}; do
        if [[ -f "$f" ]]; then
            mkdir -p "$SRC/figures/$(dirname "$fig")"
            cp "$f" "$SRC/figures/$(dirname "$fig")/"
            found=1
            break
        fi
    done
    [[ $found -eq 1 ]] || echo "warning: figure '$fig' not found in $FIGURES" | tee -a "$LOG" >&2
done
if [[ -n "${JOURNAL:-}" ]]; then
    sed -i '' -E "s/^(\\\\documentclass\[)[a-z]+,/\1${JOURNAL},/" "$SRC/$MAIN.tex"
fi
find "$SRC" \( -name .DS_Store -o -name '._*' \) -delete

# ---------- compile ----------
run_pdflatex() {
    echo "===== pdflatex pass $1 =====" >> "$LOG"
    if ! (cd "$SRC" && pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
            -output-directory "$BUILD" "$MAIN.tex") >> "$LOG" 2>&1; then
        echo "error: pdflatex pass $1 failed, see $LOG" >&2
        grep -E "^.*:[0-9]+: |^! " "$LOG" | tail -5 >&2 || true
        missing=$(grep -oE "File \`[^']+\.sty' not found" "$LOG" | sed -E "s/File \`(.*)\.sty' not found/\1/" | sort -u | tr '\n' ' ')
        [[ -n "$missing" ]] && echo "hint: tlmgr --usermode install $missing" >&2
        exit 1
    fi
}

journal=$(grep -m1 -oE '^\\documentclass\[[a-z]+' "$SRC/$MAIN.tex" | sed -E 's/.*\[//')
echo "Compiling $MAIN.tex for MDPI journal '$journal' -> $OUT"
run_pdflatex 1
echo "===== bibtex =====" >> "$LOG"
# bibtex runs in build/ but must find references.bib and Definitions/mdpi.bst in source/
# Exit status 1 means warnings only (e.g. month macros that mdpi.bst does not define); 2+ is an error.
bib_status=0
(cd "$BUILD" && BIBINPUTS="$SRC:" BSTINPUTS="$SRC:" bibtex "$MAIN") >> "$LOG" 2>&1 || bib_status=$?
if (( bib_status >= 2 )); then
    echo "error: bibtex failed, see $LOG" >&2
    grep -E "^I couldn't|^Illegal|---line" "$BUILD/$MAIN.blg" | head -5 >&2 || true
    exit 1
fi
run_pdflatex 2
run_pdflatex 3
# Rerun until cross-references are stable (at most 3 extra passes)
for pass in 4 5 6; do
    grep -q "Rerun to get" "$BUILD/$MAIN.log" || break
    run_pdflatex "$pass"
done

cp "$BUILD/$MAIN.pdf" "$OUT/"
cp "$BUILD/$MAIN.bbl" "$SRC/"   # include the .bbl so the editorial office can compile without bibtex

# ---------- submission bundle ----------
(cd "$SRC" && COPYFILE_DISABLE=1 zip -qr -X "$OUT/latex_source_${LABEL}.zip" .)
# Supplementary Materials (CSV tables + README, written by analysis/export_supplementary.py)
if [[ -d "$ROOT/supplementary" ]]; then
    (cd "$ROOT/supplementary" && COPYFILE_DISABLE=1 zip -q -X "$OUT/supplementary_materials_${LABEL}.zip" README.txt TableS*.csv)
fi

# ---------- summary ----------
undefined=$(grep -cE "undefined" "$BUILD/$MAIN.log" || true)
overfull=$(grep -cE "^Overfull" "$BUILD/$MAIN.log" || true)
pages=$(tr -d '\n' < "$BUILD/$MAIN.log" | grep -oE "Output written on [^(]*\([0-9]+ pages" | grep -oE "[0-9]+ pages" || true)
words=$(sed -E 's/\\[a-zA-Z]+(\[[^]]*\])?//g; s/[{}$~]//g' "$SRC/sections/abstract.tex" | wc -w | tr -d ' ')
echo "Done: $OUT/$MAIN.pdf (${pages:-? pages})"
echo "  undefined refs/citations: $undefined, overfull boxes: $overfull"
echo "  abstract: ~$words words (Biomedicines: about 250, structured Background/Methods/Results/Conclusions)"
# Word count from the source (citations, refs, labels and LaTeX commands removed; references excluded)
count_words() { sed -E 's/\\%/ pct/g; s/%.*//; s/\\(cite[pt]?|[cC]ref|label|ref|includegraphics)(\[[^]]*\])?\{[^}]*\}//g; s/\\[a-zA-Z]+//g' "$@" \
    | grep -oE "[A-Za-z][A-Za-z'-]*" | wc -l | tr -d ' '; }
main_files=$(grep -oE '\\input\{sections/[^}]+\}' "$SRC/$MAIN.tex" | sed -E 's/\\input\{(.*)\}/\1.tex/' | grep -v appendix)
main_words=$(cd "$SRC" && count_words $main_files)
app_words=$([[ -f "$SRC/sections/appendix.tex" ]] && count_words "$SRC/sections/appendix.tex" || echo 0)
echo "  words: main text (incl. abstract, captions, tables) ~$main_words + appendices ~$app_words = ~$((main_words + app_words)) (limit 12000, references excluded)"
echo "  full log: $LOG"
echo "  upload bundle: $OUT/latex_source_${LABEL}.zip"
