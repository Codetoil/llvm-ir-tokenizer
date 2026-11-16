from pathlib import Path
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

def file_iterator(directory):
    for path in Path(directory).rglob("*.ll"):
        yield path.read_text(encoding="utf-8")

# 1. Initialize standard BPE
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 2. Use a standard pre-tokenizer supported by llama.cpp
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# 3. Define the special tokens your offline script generated
special_tokens = [
    "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]",
    "[G]", "[COMMENT]", "[L]", "[ATTR]"
]

trainer = BpeTrainer(
    vocab_size=50_000,
    special_tokens=special_tokens,
)

# 4. Train on the pre-processed directory
print("Training tokenizer...")
tokenizer.train_from_iterator(
    file_iterator("llvm-processed"),
    trainer=trainer,
)

# 5. Save (this is now 100% GGUF compatible)
tokenizer.save("llvm-tokenizer-gguf-compatible.json")
print("Saved!")