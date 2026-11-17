import shutil

from unsloth import FastLanguageModel
import ast
import hashlib
import os
import json
from transformers import AutoTokenizer

import torch

def export_gguf():
    adapter_path = "llvm-checkpoints/final-adapter"
    export_path = "llvm-model-export"

    print("Loading fine-tuned model directly from adapter folder...")

    # 2. Load the adapter folder
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = adapter_path,
        max_seq_length = 2048,
        dtype = torch.float16,
        load_in_4bit = True,
    )

    print("Exporting to GGUF (Q4_K_M)...")
    model.save_pretrained_gguf(
        export_path,
        tokenizer,
        quantization_method = "q4_k_m"
    )

    print(f"Success! Your GGUF file is ready at: {export_path}")

def is_hash_test(test, chkhsh):
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "chkhsh"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == chkhsh
    )


def is_res_none_test(test):
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "res"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Is)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    )

def compute_hash():
    # Paths
    base_py_path = os.path.expanduser(
        "~/.unsloth/llama.cpp/conversion/base.py"
    )
    adapter_path = "llvm-checkpoints/final-adapter"

    # 1. Read and parse base.py
    with open(base_py_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)

    # 2. Extract the exact chktxt string from base.py
    chktxt = None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        if not isinstance(node.value, ast.Constant):
            continue

        if not isinstance(node.value.value, str):
            continue

        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "chktxt":
                chktxt = node.value.value

    if chktxt is None:
        raise ValueError("Could not find the 'chktxt' variable in base.py")

    print("'chktxt':", chktxt)

    # 3. Load your tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    # 4. Compute the hash
    tokens = tokenizer.encode(chktxt)
    chkhsh = hashlib.sha256(str(tokens).encode()).hexdigest()

    print("Computed hash:", chkhsh)

    # 5. Locate get_vocab_base_pre()
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "get_vocab_base_pre"
    ]

    if len(functions) != 1:
        raise RuntimeError(
            f"Expected exactly one get_vocab_base_pre(), found {len(functions)}"
        )

    function = functions[0]

    # 6. Check for an existing registration and locate the fallback
    existing_hash = False
    fallbacks = []

    for node in function.body:
        if not isinstance(node, ast.If):
            continue

        test = node.test

        if not isinstance(test, ast.Compare):
            continue

        if not isinstance(test.left, ast.Name):
            continue

        if len(test.ops) != 1 or len(test.comparators) != 1:
            continue

        comparator = test.comparators[0]

        if not isinstance(comparator, ast.Constant):
            continue

        # Existing hash registration:
        # if chkhsh == "...":
        if (
            test.left.id == "chkhsh"
            and isinstance(test.ops[0], ast.Eq)
            and comparator.value == chkhsh
        ):
            existing_hash = True

        # Fallback:
        # if res is None:
        if (
            test.left.id == "res"
            and isinstance(test.ops[0], ast.Is)
            and comparator.value is None
        ):
            fallbacks.append(node)

    if existing_hash:
        print(f"Hash {chkhsh} is already registered.")
        return

    if len(fallbacks) != 1:
        raise RuntimeError(
            f"Expected exactly one 'if res is None' fallback, "
            f"found {len(fallbacks)}"
        )

    fallback = fallbacks[0]

    # 7. Construct the registration block
    lines = source.splitlines(keepends=True)

    insert_at = fallback.lineno - 1
    original_line = lines[insert_at]

    indent = original_line[
        :len(original_line) - len(original_line.lstrip())
    ]

    newline = "\r\n" if "\r\n" in source else "\n"

    registration = (
        f'{indent}if chkhsh == "{chkhsh}":{newline}'
        f'{indent}    # NOTE: local{newline}'
        f'{indent}    res = "qwen2"{newline}'
    )

    # Preserve separation from adjacent statements.
    prefix = (
        ""
        if insert_at == 0 or not lines[insert_at - 1].strip()
        else newline
    )

    updated_source = (
        "".join(lines[:insert_at])
        + prefix
        + registration
        + newline
        + "".join(lines[insert_at:])
    )

    # 8. Validate the modified Python source
    ast.parse(updated_source)

    # 9. Back up the original file
    backup_path = base_py_path + ".bak"

    if not os.path.exists(backup_path):
        shutil.copy2(base_py_path, backup_path)
        print("Created backup:", backup_path)

    # 10. Write the modified source
    with open(base_py_path, "w", encoding="utf-8", newline="") as f:
        f.write(updated_source)

    print("Successfully inserted tokenizer registration.")
    print(f"Registered hash: {chkhsh}")


def flatten_tokenizers():
    path = "llvm-checkpoints/final-adapter/tokenizer.json"

    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)

        # Check if it's a Sequence
        if data.get("pre_tokenizer", {}).get("type") == "Sequence":
            print("Found Sequence pre-tokenizer. Flattening to ByteLevel...")
            for pt in data["pre_tokenizer"]["pretokenizers"]:
                if pt["type"] == "ByteLevel":
                    data["pre_tokenizer"] = pt
                    break

            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            print("Successfully patched tokenizer.json!")
        else:
            print("Pre-tokenizer is already not a Sequence.")
    else:
        print(f"File not found: {path}")

def deduplicate():
    path = "llvm-checkpoints/final-adapter/tokenizer.json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Map out all tokens by ID
    id_to_token = {}
    vocab = data.get("model", {}).get("vocab", {})

    if isinstance(vocab, dict):
        for tok, idx in vocab.items():
            id_to_token[idx] = tok
    elif isinstance(vocab, list):
        for idx, item in enumerate(vocab):
            id_to_token[idx] = item[0]

    # 2. Find duplicates introduced by added_tokens
    added_tokens = data.get("added_tokens", [])
    seen_strings = set(id_to_token.values())
    new_added_tokens = []

    for t in added_tokens:
        token_str = t["content"]
        token_id = t["id"]

        # If the string is already in the base vocab or another added token, skip it
        if token_str in seen_strings:
            print(f"Removing duplicate token string: '{token_str}' (ID: {token_id})")
        else:
            seen_strings.add(token_str)
            new_added_tokens.append(t)

    data["added_tokens"] = new_added_tokens

    # 3. Save the fixed tokenizer
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Token deduplication complete!")
    print(f"Final vocab size: {len(id_to_token) + len(new_added_tokens)}")

def main():
    flatten_tokenizers()
    deduplicate()
    compute_hash()
    export_gguf()

if __name__ == '__main__':
    main()