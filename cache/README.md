# `cache/` — regenerated, not shipped

MiniLM sentence embeddings are memoized here as `.npy` files keyed by a hash of
the input texts. The directory is empty in the released archive because its
contents are derived and large; they are rebuilt automatically on first use.

With `sentence-transformers` installed (see `../requirements.txt`) the cache is
filled by `run_experiment.py` and by the `revision/` scripts. Without it, the
revision scripts fall back to the equivalent mean-pooling implementation in
`revision/rev_common.py` (`encode_minilm`), which agrees with the cached
encoder to 3e-7.

Deleting this directory costs runtime only; it never changes a result.
