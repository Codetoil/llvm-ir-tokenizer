import json
import os

def main():
    path = os.path.expanduser("~/cluster1-storage/llvm-checkpoints/final-adapter/tokenizer.json")

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

if __name__ == '__main__':
    main()