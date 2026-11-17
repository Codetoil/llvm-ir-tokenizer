from transformers import AutoTokenizer
from tokenizers import Tokenizer, AddedToken

def main():
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

if __name__ == '__main__':
    main()