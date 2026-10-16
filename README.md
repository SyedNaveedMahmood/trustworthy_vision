# Trustworthy Vision: Controlled Visual Evidence Pilot

Status: DESIGN ONLY. No models have been run, datasets downloaded, or experimental claims verified.

Objective: test whether a controlled, artifact-aware, operator-stable intervention score is a better measure of frozen vision-language model (VLM) reliance on task-relevant visual evidence than raw answer sensitivity to a single masked image.

Hardware: one NVIDIA RTX 4080 SUPER, 16 GB VRAM. Two frozen, independently trained, open-weight VLM families, loaded one at a time. No model fine-tuning or training of a new detector in the pilot.

## Read in this order

1. [Decision and claims](docs/01_RESEARCH_QUESTION_AND_HYPOTHESES.md): the proposed gap and the close-prior constraints.
2. [Data, tasks, splits](docs/02_DATA_AND_SPLITS.md): GQA VQA + RefCOCOg referring-expression selection.
3. [Interventions and controls](docs/03_INTERVENTION_DESIGN.md): exactly what edits to make and how to validate them.
4. [Experiments](docs/04_EXPERIMENT_MATRIX.md): six staged studies, outcomes, ablations, failure criteria.
5. [Metrics and statistical analysis](docs/05_MEASUREMENT_AND_STATISTICS.md): conditional choice likelihood, tests, blind holdout.
6. [Prior work, repos, licenses](docs/06_REPRODUCIBILITY_AND_PRIOR_ART.md): which official repositories can actually be reused.
7. [GPU and stop gates](docs/07_GPU_EXECUTION_AND_GATES.md): 16 GB VRAM budget and reproducibility.
8. [Implementation request](CODING_AGENT_MASTER_PROMPT.md): the prompt to give your coding agent.
9. [Machine-readable protocol](configs/pilot_v1.yaml).

## Principle

A model's changed answer after image masking shows sensitivity to the edited input. It does **not**, by itself, prove causal reliance on the intended semantic evidence. Editing artifacts, other removed context, prompt changes, region size and option-label bias can create similar effects.

The project is falsification-first: show the confound empirically, measure the control quality, compare fairly to recent papers, and stop if the proposed correction does not improve on strong controls.

## Scope and status

This is an executable *design*, not a completed experiment. Ground-truth evidence masks are used only for controlled diagnostic evaluation. A general-purpose localization claim would require a separate no-oracle region-proposal experiment. Paper acceptance, code availability and licenses must be freshly audited before execution.

Output policy: commit code, small configs, unit tests, experiment manifests and summaries; do not commit dataset images, medical images, model weights, token credentials or multi-GB artifacts. Reuse repositories as pinned third-party references rather than silently vendoring them.

The first output a coding agent must produce is the source audit and a running smoke test, not a result table invented from expectations.
