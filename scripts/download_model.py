"""Optional one-time download of MIT-licensed multilingual E5-small.

Pinning an immutable model revision prevents silent changes across experiments.
Inference itself uses local_files_only=True and never downloads weights.
"""
import json
from pathlib import Path
import subprocess

REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "artifacts/models/multilingual-e5-small"
FILES = ["config.json", "modules.json", "sentence_bert_config.json", "special_tokens_map.json",
         "tokenizer.json", "tokenizer_config.json", "1_Pooling/config.json", "model.safetensors"]


def main():
    for filename in FILES:
        path = TARGET / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".partial")
            subprocess.run(["curl", "-fL", "--retry", "3",
                f"https://huggingface.co/intfloat/multilingual-e5-small/resolve/{REVISION}/{filename}",
                "-o", str(temporary)], check=True)
            temporary.replace(path)
    (TARGET / "provenance.json").write_text(json.dumps({"model": "intfloat/multilingual-e5-small",
        "revision": REVISION, "license": "MIT", "source": "https://huggingface.co/intfloat/multilingual-e5-small"}, indent=2))
    print(TARGET)


if __name__ == "__main__":
    main()
