# Third-party integrations

See [SOURCE_AUDIT.md](SOURCE_AUDIT.md) for verified upstream URLs, license limitations, immutable source/model revisions, actual execution status and baseline compatibility. Weight SHA-256 values are in `reports/evidence/weight_checksums.json` and `lama_checkpoint.json`.

The official LaMa checkout is locally ignored in `repos/lama`. Primary models reside in the local Hugging Face cache; LaMa weights are ignored in `third_party_weights`. No third-party code, model weights or images are vendored into Git. Both primary models use official Transformers 4.51.3, BF16 without quantization, in separate processes. LaMa uses a separate Python 3.8 CPU environment and unmodified official `bin/predict.py`.
