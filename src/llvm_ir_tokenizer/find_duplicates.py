import json
from collections import Counter
import os

def main():
    path = os.path.expanduser("llvm-checkpoints/final-adapter/tokenizer.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = data.get("model", {}).get("vocab", {})
    # If vocab is a list of [token, score] pairs (Unigram), extract the tokens
    if isinstance(vocab, list):
        tokens = [v[0] for v in vocab]
    else:
        tokens = list(vocab.keys())

    duplicates = [token for token, count in Counter(tokens).items() if count > 1]

    print("Total tokens:", len(tokens))
    print("Duplicate tokens found:", len(duplicates))
    for dup in duplicates:
        print(f"  - '{dup}' (appears {Counter(tokens)[dup]} times)")

if __name__ == '__main__':
    main()