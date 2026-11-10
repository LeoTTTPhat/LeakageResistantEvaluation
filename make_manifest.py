#!/usr/bin/env python3
"""Create a machine-readable checksum manifest for generated result files."""

from __future__ import annotations

import json

from download_data import HERE, sha256


RESULTS = HERE / "results"


def main() -> None:
    entries = []
    for path in sorted(RESULTS.iterdir()):
        if not path.is_file() or path.name == "manifest.json":
            continue
        entries.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    (RESULTS / "manifest.json").write_text(
        json.dumps({"files": entries}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote checksums for {len(entries)} result files.")


if __name__ == "__main__":
    main()
