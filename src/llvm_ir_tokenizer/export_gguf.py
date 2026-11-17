from unsloth import FastLanguageModel
import os

import torch

def main():
    # 1. Define absolute paths
    home = os.path.expanduser("~")
    adapter_path = os.path.join(home, "cluster1-storage", "llvm-checkpoints", "final-adapter")
    export_path = os.path.join(home, "cluster1-storage", "llvm-model-export")

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

if __name__ == '__main__':
    main()