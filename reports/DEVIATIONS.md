# Deviations and unresolved protocol issues

Recorded before any confirmatory scoring, 2026-10-16 (Asia/Dhaka).

1. Native Windows is used for the initial VLM smoke. WSL has no installed distribution configured for execution. No driver changes were made. CUDA and BF16 must be measured before claiming compatibility.
2. The initially empty workspace lacked Git and Python. Portable Git and uv-managed Python 3.11 were installed outside the repository, with an isolated repository virtual environment.
3. No humans are available to perform the two-rater semantic edit audit. Automated shape/geometry checks cannot satisfy semantic validity: HUMAN_AUDIT_PENDING.
4. **Unresolved control construction:** docs/05 requires label-free off-target sampling for each candidate; docs/03 describes the oracle negative candidate as the off-target control in a 13-image factorial. These are not equivalent. A negative candidate chosen by relevance labels must not become an inference-time control. Swapping the positive and negative as mutual controls gives z_minus = -z_plus, equal MAD penalties, and makes lambda irrelevant to pair rankings. A shared independent label-free control needs additional edits unless a justified design reuses it. Do not lock confirmatory evaluation until this is explicitly resolved and costed on development data.
5. No new benchmark, substitute annotations, fake LaMa output or invented literature baseline will be used to fill missing inputs. Synthetic shapes are infrastructure diagnostics only.
