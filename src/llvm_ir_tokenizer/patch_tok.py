import json
import os

# Use the absolute path to avoid the tilde error
path = "/home/codetoil/cluster1-storage/llvm-checkpoints/final-adapter/tokenizer.json"

if os.path.exists(path):
    with open(path, "r") as f:
        data = json.load(f)

    # Check if it's a Sequence
    if data.get("pre_tokenizer", {}).get("type") == "Sequence":
        print("Found Sequence pre-tokenizer. Flattening to ByteLevel...")
        for pt in data["pre_tokenizer"]["pretokenizers"]:
            if pt["type"] == "ByteLevel":
                data["pre_tokenizer"] = pt
                break
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print("Successfully patched tokenizer.json!")
    else:
        print("Pre-tokenizer is already not a Sequence.")
else:
    print(f"File not found: {path}")
