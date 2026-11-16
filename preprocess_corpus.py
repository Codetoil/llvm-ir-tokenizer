import re
import json
from pathlib import Path
from typing import List, Tuple, Dict

from pygments.lexers.asm import LlvmLexer
from pygments.token import Token as PToken

LOCAL_RE = re.compile(r'^%[a-zA-Z$._][\w$.-]*$|^%\d+$')
GLOBAL_RE = re.compile(r'^@[a-zA-Z$._][\w$.-]*$|^@\d+$')
LABEL_RE = re.compile(r'^[a-zA-Z$._][\w$.-]*:$|^\d+:$')
ATTR_RE = re.compile(r'^#\d+$')

INLINE_COMMENT_PREFIXES = ("; RUN:", "; XFAIL:", "; REQUIRES:")


class LLVMPreProcess:
    def __init__(self):
        self.lexer = LlvmLexer()

    def preprocess(self, text: str) -> Tuple[List[Tuple[str, Tuple[int, int]]], Dict, Dict, Dict, Dict]:
        raw = list(self.lexer.get_tokens_unprocessed(text))

        # Forward and reverse maps for locals and labels
        local_to_id, id_to_local = {}, {}
        label_to_id, id_to_label = {}, {}

        # Reverse maps for comments and globals
        comment_to_id, id_to_comment = {}, {}
        global_to_id, id_to_global = {}, {}

        # Pass 1: Build maps by definition order
        for _, ttype, value in raw:
            if ttype in PToken.Comment:
                continue

            if LOCAL_RE.match(value) and value not in local_to_id:
                local_idx = len(local_to_id)
                local_to_id[value] = f"%{local_idx}"
                id_to_local[str(local_idx)] = value

            elif LABEL_RE.match(value) and value not in label_to_id:
                label_idx = len(label_to_id)
                label_to_id[value] = f"bb{label_idx}:"
                id_to_label[str(label_idx)] = value

        # Pass 2: Generate the output tokens and spans
        out = []
        for idx, ttype, value in raw:
            span = (idx, idx + len(value))
            if not value:
                continue

            # 1. Handle Text (Whitespace) FIRST to prevent dropping newlines
            if ttype in PToken.Text:
                for i, c in enumerate(value):
                    if c == "\n":
                        out.append(("\n", (idx + i, idx + i + 1)))
                continue

            # 2. Extract trailing newline for all other token types
            has_newline = value.endswith("\n")
            if has_newline:
                value = value[:-1]
                span = (span[0], span[1] - 1)
                if not value:
                    out.append(("\n", (span[1], span[1] + 1)))
                    continue

            # 3. Comments -> [COMMENT] <id>
            if ttype in PToken.Comment:
                stripped = value.strip()
                if not stripped.startswith(INLINE_COMMENT_PREFIXES):
                    if stripped not in comment_to_id:
                        c_id = len(comment_to_id)
                        comment_to_id[stripped] = c_id
                        id_to_comment[str(c_id)] = stripped

                    out.append(("[COMMENT]", (idx, idx + 1)))
                    out.append((str(comment_to_id[stripped]), span))
                else:
                    cursor = 0
                    for word in value.split():
                        p = value.find(word, cursor)
                        out.append((word, (idx + p, idx + p + len(word))))
                        cursor = p + len(word)

            # 4. Locals -> %0, %1, ...
            elif LOCAL_RE.match(value):
                out.append((local_to_id[value], span))

            # 5. Labels -> [L] bb0:
            elif LABEL_RE.match(value):
                out.append(("[L]", (idx, idx + 1)))
                out.append((label_to_id[value], span))

            # 6. Globals -> [G] <id>
            elif GLOBAL_RE.match(value):
                if value not in global_to_id:
                    g_id = len(global_to_id)
                    global_to_id[value] = g_id
                    id_to_global[str(g_id)] = value

                out.append(("[G]", (idx, idx + 1)))
                out.append((str(global_to_id[value]), span))

            # 7. Attr groups -> [ATTR] 0
            elif ATTR_RE.match(value):
                out.append(("[ATTR]", (idx, idx + 1)))
                out.append((value[1:], (idx + 1, idx + len(value))))

            # 8. Unchanged tokens
            else:
                out.append((value, span))

            # Re-append the newline we extracted in step 2
            if has_newline:
                out.append(("\n", (span[1], span[1] + 1)))

        return out, id_to_comment, id_to_global, id_to_local, id_to_label


def process_file(file_path: Path, input_dir: Path, output_dir: Path):
    text = file_path.read_text(encoding="utf-8")
    preprocessor = LLVMPreProcess()

    atoms, comments, globals_, locals_, labels_ = preprocessor.preprocess(text)

    # Join atoms with spaces, then use regex to clean up spaces around newlines
    processed_text = " ".join(atom for atom, _ in atoms)
    processed_text = re.sub(r' *\n *', '\n', processed_text)

    relative_path = file_path.relative_to(input_dir)
    out_path = output_dir / relative_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(processed_text, encoding="utf-8")

    # Write the complete side-channel mapping
    json_path = out_path.with_suffix(".ll.json")
    mapping = {
        "comments": comments,
        "globals": globals_,
        "locals": locals_,
        "labels": labels_
    }
    json_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")


if __name__ == "__main__":
    input_dir = Path("llvm-project")
    output_dir = Path("llvm-processed")

    output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in input_dir.rglob("*.ll"):
        process_file(file_path, input_dir, output_dir)
        print(f"Processed {file_path}")