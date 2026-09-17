# Research Lab Roadmap

This repository studies parameter-efficient fine-tuning (PEFT), beginning with the upstream Microsoft LoRA implementation.

## Phases

- [x] Import and preserve the upstream LoRA baseline
- [x] Add an isolated research-lab structure
- [ ] Add reproducible experiment configuration
- [ ] Add trainable-parameter and memory reporting
- [ ] Add LoRA vs full fine-tuning comparison
- [ ] Add benchmark reports and reproducibility checks

## Principles

1. Keep upstream code attributable and minimally modified.
2. Separate original experiments from upstream examples.
3. Record configuration, metrics, environment, and hardware for every run.
4. Never report benchmark results without executing the experiment.
