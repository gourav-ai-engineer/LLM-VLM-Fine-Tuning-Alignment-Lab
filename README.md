# LLM/VLM Fine-Tuning & Alignment Lab

A reproducible research and engineering laboratory for parameter-efficient fine-tuning, multimodal adaptation, alignment experiments, and evaluation.

> This repository is an independent research lab built on open-source ideas and references. Upstream code and papers are documented in `UPSTREAM.md`; original contributions are developed under `src/` and `experiments/`.

## Research Tracks

- **PEFT:** LoRA/QLoRA configuration and trainable-parameter analysis
- **Data quality:** validation, deduplication, leakage checks, and dataset reports
- **Evaluation:** task metrics, regression checks, and experiment comparisons
- **Alignment:** controlled preference-optimization experiments
- **VLM:** image-text dataset contracts and multimodal experiment interfaces
- **Reproducibility:** configuration-driven runs, deterministic seeds, and JSON artifacts

## Repository Layout

```text
src/lab/
  config.py          # typed experiment configuration
  data.py            # dataset validation primitives
  metrics.py         # parameter and metric utilities
  reproducibility.py # seed and run metadata helpers
experiments/
  configs/           # versioned experiment configurations
  reports/           # generated reports (ignored by git)
tests/
```

## Quick start

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -q
```

The first milestone uses lightweight, dependency-minimal utilities so the project can be tested on CPU before adding GPU-heavy training dependencies.

## Engineering principles

1. Establish a measurable baseline before optimizing.
2. Keep data, configuration, code, and generated artifacts separate.
3. Never report benchmark improvements without a recorded baseline.
4. Record model, dataset, seed, hardware, and software versions for each run.
5. Treat safety and dataset provenance as first-class experiment metadata.

## Roadmap

- [x] Typed experiment configuration
- [x] Dataset contract and validation report
- [x] Trainable-parameter accounting utilities
- [ ] Baseline SFT runner
- [ ] LoRA/QLoRA adapter runner
- [ ] Evaluation harness and regression gates
- [ ] Preference optimization track
- [ ] Vision-language experiment track
- [ ] Experiment dashboard
