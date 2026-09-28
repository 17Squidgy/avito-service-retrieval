"""Download the organizer's public archive; verify it before extracting.

Network access is used only during setup. Training and inference are offline.
"""
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import urlencode
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/raw"
PUBLIC_KEY = "https://disk.yandex.ru/d/sNhfo0YOjGtufg"
SHA256 = "8dd3cba59201bae333c11db89c5111198fa10cc52a70c70bdc57bd6a248fb777"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    archive = DEST / "dataset.zip"
    if not archive.exists():
        endpoint = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
        response = subprocess.check_output([
            "curl", "-fsSL", endpoint + urlencode({"public_key": PUBLIC_KEY, "path": "/dataset.zip"})
        ])
        subprocess.run(["curl", "-fL", "--retry", "3", json.loads(response)["href"],
                        "-o", str(archive)], check=True)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != SHA256:
        raise ValueError(f"Archive checksum mismatch: {digest}; remove incomplete archive and retry")
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            if member.filename.endswith(".parquet"):
                # Flatten only the three expected files, never arbitrary archive paths.
                name = Path(member.filename).name
                if name in {"train.parquet", "benchmark_queries.parquet", "benchmark_items.parquet"}:
                    target = DEST / name
                    with zipped.open(member) as source, target.open("wb") as output:
                        import shutil
                        shutil.copyfileobj(source, output)
                    print(target)


if __name__ == "__main__":
    main()
