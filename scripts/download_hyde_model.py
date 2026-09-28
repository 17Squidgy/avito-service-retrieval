"""Download a pinned Apache-2.0 Qwen model for offline HyDE generation."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
TARGET = Path(__file__).resolve().parents[1] / "artifacts/models/qwen2.5-1.5b-instruct"
FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
         "merges.txt", "vocab.json", "LICENSE", "model.safetensors"]


def download(filename):
    target = TARGET / filename
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".partial")
        subprocess.run(["curl", "-fsSL", "--retry", "3",
            f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{filename}", "-o", str(temporary)], check=True)
        temporary.replace(target)
    print(filename, flush=True)


if __name__ == "__main__":
    TARGET.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(download, FILES))
    (TARGET / "provenance.json").write_text(json.dumps({"model": MODEL, "revision": REVISION,
                                                      "license": "Apache-2.0"}, indent=2))
