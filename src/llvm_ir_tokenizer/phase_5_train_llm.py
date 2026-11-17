import os

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
from pathlib import Path

def main():
    # 1. Configuration
    max_seq_length = 2048 # 2048 is safe for 12GB VRAM. Increase to 4096 if you don't OOM.
    dtype = torch.float16
    load_in_4bit = True
    output_dir = "llvm-checkpoints"

    # 2. Load Base Model in 4-bit
    print("Loading base Qwen2.5-Coder-7B model...")
    model, _ = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # 3. Load YOUR Custom Tokenizer
    print("Loading custom merged tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("./qwen-llvm-merged-tokenizer")
    # Ensure the tokenizer has a padding token and knows which side to pad on
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    torch.cuda.empty_cache()
    print(f"Resizing embeddings to match tokenizer vocab size: {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    # 5. Add LoRA Adapters (only training ~1-2% of the model)
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth", # Huge VRAM saver
        random_state = 3407,
    )

    # 6. Load the Preprocessed Dataset
    print("Loading dataset from llvm-processed...")
    # This grabs all .ll files from your preprocessed folder
    ll_files = [str(p) for p in Path("llvm-processed").rglob("*.ll")]
    dataset = load_dataset("text", data_files={"train": ll_files}, split="train")

    # 7. Training Arguments
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        args = TrainingArguments(
            per_device_train_batch_size = 2,   # Keep at 2 for 12GB VRAM
            gradient_accumulation_steps = 4,   # Effective batch size = 8
            warmup_steps = 5,
            max_steps = 200,                   # Adjust based on how many files you have
            learning_rate = 2e-4,
            fp16 = True,
            bf16 = False,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = output_dir,           # Saves to cluster1-storage!
            save_steps = 50,
            torch_compile=False
        ),
    )

    # 8. Start Training
    print("Starting training...")
    trainer.train()

    # 9. Save the final LoRA adapter
    print("Saving LoRA adapter...")
    model.save_pretrained(f"{output_dir}/final-adapter")
    tokenizer.save_pretrained(f"{output_dir}/final-adapter")
    print("Done!")

if __name__ == '__main__':
    main()