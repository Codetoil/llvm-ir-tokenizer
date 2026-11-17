from tokenizers import Tokenizer, AddedToken
from transformers import AutoTokenizer
import json

def merge_qwen():
    # 1. Load the base Qwen tokenizer
    # (Make sure you have the Qwen2.5 tokenizer downloaded)
    qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

    # 2. Load our custom LLVM tokenizer
    llvm_tok = Tokenizer.from_file("llvm-code-tokenizer.json")
    llvm_vocab = list(llvm_tok.get_vocab().keys())

    # 3. Filter out standard tokens to only get the custom LLVM ones
    # (The first 15000 tokens will include byte-level fallbacks, we want the specialized ones)
    # Note: It's easier to just add all of them and let HF handle duplicates safely.
    llvm_tokens = [tok for tok in llvm_vocab if tok not in ["[UNK]", "[G]", "[COMMENT]", "[L]", "[ATTR]"]]

    # 4. Define the channel tags
    channel_tokens = [
        "<|think|>", "<|end_think|>",
        "<|symbol|>", "<|end_symbol|>",
        "<|code|>", "<|end_code|>",
        "<|global|>", "<|local|>", "<|label|>", "<|comment|>"
    ]

    # 5. Create AddedToken objects (normalized=False is CRITICAL)
    # This prevents Qwen's regex from splitting "alloca" into "al" + "loca"
    added_tokens = [
        AddedToken(t, normalized=False, special=False) for t in llvm_tokens
    ]
    added_tokens.extend([
        AddedToken(t, normalized=False, special=True) for t in channel_tokens
    ])
    added_tokens.extend([
        AddedToken(t, normalized=False, special=True) for t in ["[UNK]", "[G]", "[COMMENT]", "[L]", "[ATTR]"]
    ])

    # 6. Add them to Qwen
    num_added = qwen_tok.add_tokens(added_tokens)
    print(f"Added {num_added} new tokens to Qwen.")

    # 7. Save the merged tokenizer
    qwen_tok.save_pretrained("qwen-llvm-merged-tokenizer")

def deduplicate_qwen():
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
    merge_qwen()
    deduplicate_qwen()

if __name__ == '__main__':
    main()
