import re
from pathlib import Path
from tokenizers import Tokenizer, pre_tokenizers, Regex, AddedToken
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer


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

def main():
    train_tokenizer()

if __name__ == '__main__':
    main()
