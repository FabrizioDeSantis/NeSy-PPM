"""
Remove the unresolved merge-conflict blocks committed to NeSy-PPM, keeping the HEAD side.

The HEAD side of every block is the refactored code that main.py and nesy/ use
(e.g. LSTMModel(feature_names, vocab_sizes, seq_len, ...), common.preprocess_eventlog);
the other side (commit 4f36cb9...) is the pre-refactoring code.

    python fix_conflicts.py /path/to/NeSy-PPM          # dry run: list the blocks
    python fix_conflicts.py /path/to/NeSy-PPM --apply  # rewrite the files in place

Afterwards: python -m compileall -q model data nesy main.py  (must print nothing),
regenerate knowledge_base.txt with make_knowledge_base.py, and commit.
"""
import argparse
import pathlib
import re

START, MID, END = re.compile(r"^<{7}( |$)"), re.compile(r"^={7}$"), re.compile(r"^>{7}( |$)")


def resolve(lines):
    out, state, blocks = [], "plain", 0
    for line in lines:
        bare = line.rstrip("\n")
        if state == "plain" and START.match(bare):
            state, blocks = "head", blocks + 1
        elif state == "head" and MID.match(bare):
            state = "theirs"
        elif state == "theirs" and END.match(bare):
            state = "plain"
        elif state in ("plain", "head"):
            out.append(line)
    if state != "plain":
        raise ValueError("unterminated conflict block")
    return out, blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = pathlib.Path(args.repo)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts or path.suffix not in {".py", ".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        if not any(START.match(l.rstrip("\n")) for l in text):
            continue
        resolved, blocks = resolve(text)
        print(f"{path.relative_to(root)}: {blocks} conflict block(s){' -> fixed' if args.apply else ''}")
        if args.apply:
            path.write_text("".join(resolved), encoding="utf-8")


if __name__ == "__main__":
    main()
