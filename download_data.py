#!/usr/bin/env python3
"""Download the two public corpora used by the ESWA robustness study."""

from __future__ import annotations

import hashlib
import pathlib
import ssl
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"

FILES = {
    "phishfuzzer.parquet": (
        "https://huggingface.co/datasets/"
        "simlab-vs/PhishFuzzer_emails_entity_rephrased_v1/resolve/"
        "839d0d927d3e6a39e2b007a3c71b8fc414213f2e/"
        "data/train-00000-of-00001.parquet",
        "c5662d96f3c861623a1c0b3c8383dc23aac442ccba99d9d4c5be4d9f9822f6c2",
    ),
    "gpt_phishing_email.csv": (
        "https://huggingface.co/datasets/"
        "Dizzzy0x00/LLMGen-Phishing-Email-Dataset/resolve/"
        "d717e5457b71cb44dd038dafe6a81cea5e0badab/"
        "GPT_Phishing_Email_dataset.csv",
        "e3a335c8acacec09afaa4b9a14c7010195063310e4dcfbae38c1fa50b8d48dc3",
    ),
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    import certifi

    DATA.mkdir(parents=True, exist_ok=True)
    for filename, (url, expected) in FILES.items():
        destination = DATA / filename
        if not destination.exists():
            print(f"Downloading {filename} ...", flush=True)
            context = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(url, context=context) as response:
                destination.write_bytes(response.read())
        actual = sha256(destination)
        if actual != expected:
            raise RuntimeError(
                f"Checksum mismatch for {filename}: expected {expected}, got {actual}"
            )
        print(f"{filename}: verified ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
