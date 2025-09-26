# Optional SFT Model Deployment Notes

This repo now loads SFT-enhanced models only if they are present. The
following outlines how to distribute or download them in a future release.

## Expected locations

* Chunk semantics classifier: `artifacts/models/chunk_classifier_sft/`
  (files: `config.json`, `model.safetensors`, tokenizer artifacts).
* Reranker: `artifacts/models/reranker_sft/` (SentenceTransformers directory).

At runtime the loader checks those directories; if they are missing the
pipeline falls back to heuristic behaviour.

## Possible download flow (future work)

1. Host the model folders as compressed archives (e.g. `.tar.gz`) on a CDN or
   Hugging Face Space. Keep checksums alongside the downloads.
2. Add a CLI helper (e.g. `python -m tools.install_models --with-sft`) that:
   * Prompts the user that ~600 MB will be downloaded.
   * Streams each archive, verifies the checksum, and extracts it to the
     locations above.
   * Writes a marker file (`.installed`) so repeat runs skip unless `--force`.
3. In the GUI/settings, expose a toggle or button that invokes the same helper
   (with progress UI).
4. Document environment overrides (`CHUNK_SEM_MODEL`, `RERANKER_LOCAL_PATH`) so
   advanced users can point to custom fine-tunes.

## Removal / cleanup script idea

Add `python -m tools.install_models --remove-sft` to delete the folders and
free disk space.

These steps keep the default install light, while letting power users opt into
larger SFT packages on demand.

## Web search providers

Set these options in `config.local.json` (or similar) so users do not need to
export environment variables:

```json
{
  "WEB_SEARCH_PROVIDER": "serpapi",
  "SERPAPI_KEY": "sk-..."
}
```

Supported values:

- `serpapi` (requires `SERPAPI_KEY`, free tier ≈100 queries/month)
- `brave` (requires `BRAVE_API_KEY`, free tier ≈2000 queries/month)
- anything else defaults to DuckDuckGo (no key)

The runtime will still honour environment variables if present, but config
values take precedence.
