# `data/` — downloaded, never redistributed

This directory is intentionally empty in the released archive. The artifact
does not redistribute the upstream corpora; it downloads them and verifies
them.

Populate it with:

```bash
python3.11 ../download_data.py
```

which fetches both files at pinned upstream revisions and checks each against a
SHA-256 recorded in `download_data.py`:

| File | Source | Pinned revision | Licence |
| --- | --- | --- | --- |
| `phishfuzzer.parquet` | `simlab-vs/PhishFuzzer_emails_entity_rephrased_v1` | `839d0d92` | none declared upstream |
| `gpt_phishing_email.csv` | `Dizzzy0x00/LLMGen-Phishing-Email-Dataset` | `d717e545` | Apache-2.0 |

The PhishFuzzer repository declared no software or data licence at the pinned
revision, which is why this artifact downloads rather than redistributes it.

`verify_artifact.py` fails if either file is absent or its checksum differs, so
run `download_data.py` before verification.
