import re
from pathlib import Path
from tokenizers import Tokenizer, pre_tokenizers, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

def code_only_iterator(directory):
    """
    Strips the specific text from symbol blocks, leaving only the structural tags.
    Example: '<|symbol|> my_memcpy <|global|> 3 <|end_symbol|>'
    Becomes: '<|symbol|><|global|><|end_symbol|>'
    """
    for path in Path(directory).rglob("*.ll"):
        text = path.read_text(encoding="utf-8")

        # 1. Collapse the symbol blocks to just their structural tags
        # Matches: <|symbol|> [text] <|global|> [id] <|end_symbol|>
        text = re.sub(
            r'(<\|symbol\|>).*?(<\|(?:global|local|label|comment)\|>).*?(<\|end_symbol\|>)',
            r'\1\2\3',
            text,
            flags=re.DOTALL
        )

        # 2. Remove the code wrapper tags (optional, but keeps vocab clean)
        text = text.replace("<|code|>", "").replace("<|end_code|>", "")

        if text.strip():
            yield text

def train_tokenizer():
    # 1. Initialize standard BPE
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # 2. Pre-tokenizer Sequence
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        # Isolate all structural tags and LLVM symbols
        pre_tokenizers.Split(
            Regex(
                r"(\[G\]|\[COMMENT\]|\[L\]|\[ATTR\]|<\|symbol\|>|<\|end_symbol\|>|<\|global\|>|<\|local\|>|<\|label\|>|<\|comment\|>)"),
            behavior="isolated"
        ),
        # Apply ByteLevel to the remaining code text
        pre_tokenizers.ByteLevel(add_prefix_space=False)
    ])

    # 3. Define the special tokens
    # We MUST include the structural tags here so BPE learns them as atomic tokens
    special_tokens = [
        "[UNK]", "[G]", "[COMMENT]", "[L]", "[ATTR]",
        "<|symbol|>", "<|end_symbol|>",
        "<|global|>", "<|local|>", "<|label|>", "<|comment|>"
    ]

    trainer = BpeTrainer(
        vocab_size=15_000,
        special_tokens=special_tokens,
    )

    print("Training custom LLVM tokenizer on structural code...")
    tokenizer.train_from_iterator(
        code_only_iterator("llvm-processed"),
        trainer=trainer,
    )

    tokenizer.save("llvm-code-tokenizer.json")
    print("Saved llvm-code-tokenizer.json")

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

def main():
    train_tokenizer()
    merge_qwen()

if __name__ == '__main__':
    main()