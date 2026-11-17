import ast
import hashlib
import os
from transformers import AutoTokenizer

def main():
    # Paths
    base_py_path = os.path.expanduser("~/.unsloth/llama.cpp/conversion/base.py")
    adapter_path = os.getcwd() + "llvm-checkpoints/final-adapter"

    # 1. Extract the exact `chktxt` string from base.py
    chktxt = None
    with open(base_py_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "chktxt":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        chktxt = node.value.value

    if not chktxt:
        raise ValueError("Could not find the 'chktxt' variable in base.py")
    print("'chktxt': ", chktxt)

    # 2. Load your tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    # 3. Compute the hash
    tokens = tokenizer.encode(chktxt)
    chkhsh = hashlib.sha256(str(tokens).encode()).hexdigest()

    # 4. Output the block
    print("\n=== COPY THE FOLLOWING BLOCK ===\n")
    print(f'        if chkhsh == "{chkhsh}":')
    print('            # NOTE: local')
    print('            res = "llvm-ir-qwen"')
    print("\n================================")
if __name__ == '__main__':
    main()