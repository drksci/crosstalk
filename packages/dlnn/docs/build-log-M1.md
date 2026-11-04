# Build log M1 (trained ternary TDL-Net-S)

Files: `dlnn/model/{ternary,tdlnet_torch,export,train}.py`, `tests/test_model.py`, `docs/model_export.md`,
outputs `models/tdlnet_modrec_{ternary,float}.npz` (+ `.pt` state dicts), `models/train_report.json`.
Extra (not required): `models/tdlnet_modrec_w32_{ternary,float}.npz`, `models/train_report_w32.json` (hidden width 32).
Installed: `torch 2.15.0+cpu` (and its deps filelock, fsspec, mpmath, networkx, setuptools 78.1.0, sympy) into `.venv`
via `pip install torch --index-url https://download.pytorch.org/whl/cpu`. `torch.set_num_threads(2)` is set in `train.py`.

## Commands and actual results

```
.venv/bin/python -m dlnn.model.train --task modrec          # total 202 s (box shared with other workers; 3 restarts per model)
  ternary: acc 0.7348  acc>=10dB 0.7629  per-SNR {'0': 0.62875, '5': 0.75625, '10': 0.75, '15': 0.7675, '20': 0.77125}  (116.9s)
  float:   acc 0.8465  acc>=10dB 0.8746  per-SNR {'0': 0.76625, '5': 0.8425, '10': 0.86875, '15': 0.89, '20': 0.865}  (83.6s)
.venv/bin/python -m pytest -q tests/test_model.py            # 8 passed
```
(run twice with identical numbers: the CLI is deterministic.)
Logit reproduction on the first 256 test examples, shipped models, `dlnn.tiers.virtual.forward` (numpy) vs torch eval:
ternary max abs error 1.9e-6 (logit range +-17); float 3.8e-6. Tier V on the full test set = 0.73475 (equals torch);
tier E (`emulated.forward`, unmodified) accuracy 0.73475, top-1 agreement E vs V 0.99575.
Ternary model: 1152 conv taps (128+512+512) + 64 ternary readout weights; non-zero tap fraction 0.65/0.53/0.42;
vsat per layer 1.297 / 1.203 / 2.077 (readout values +-4.13).
Per class (ternary): OOK .934 BPSK .912 QPSK .632 8PSK **0.0** 16QAM .490 2FSK .990 4FSK .994 GMSK .926.
Per class (float): OOK .990 BPSK .966 QPSK .534 8PSK .454 16QAM .844 2FSK .998 4FSK .996 GMSK .990.

## Acceptance status

| criterion | target | result |
|---|---|---|
| train time | <= 10 min | 3.4 min (both models, 3 seeds each) |
| float mixed-SNR | >= 80 % | **84.65 %** pass |
| ternary mixed-SNR | >= 70 % | **73.48 %** pass |
| ternary at SNR >= 10 dB | >= 85 % | **76.29 %: NOT MET** |
| V reproduces torch logits | <= 1e-4 | 1.9e-6 pass |
| pytest | pass | pass |

## Why the 85 % (>= 10 dB) ternary target is not met, and what was tried

Evidence that this is a capacity limit of the width-8 ternary topology, not an optimisation bug: with the *same code*,
width 16 ternary reached 81.0 % mixed (83.8 % at >= 10 dB, validation split), width 32 reached 86.3 % mixed /
**89.1 % at >= 10 dB on the test set** (`--width 32`, 1 seed; float 91.9 %). Width 8 float itself only reaches 87.5 % at >= 10 dB.
The ternary S net drops 8PSK completely (0 %), the hardest class (needs 8th-order phase statistics under CFO).

Tried (validation = last 2000 of the 16k train; mixed-SNR val accuracy; ternary unless noted):

| change | result |
|---|---|
| plan recipe lr 3e-3, 20 epochs (first model version, test set) | float 62.1 %, ternary 45.1 % (underfit) |
| float, 20 epochs lr 3e-3 / 1e-2; 60 ep lr 1e-2 / 3e-2; 100 ep lr 1e-2 | 64.5 / 71.5; 78.6 / 82.1 / 81.2 |
| float input_scale 0.3 / 3 / 2 (20-60 ep) | 42.8 / 62.1 / 81.0 |
| float without rotation aug | 64.3 (aug vs not: ~equal; train acc ~ val acc, i.e. underfitting) |
| ternary 60 ep lr 3e-2 / 1e-2 / 1e-1 | 71.0 / 67.4 / 67.4 |
| ternary 150 epochs lr 3e-2 | 67.0 (no gain from more epochs) |
| ternary input_scale 2 | 67.2 |
| per-output-line scale (post-tanh gain per line, plus per-class readout scale) | 70.9 / 72.3 (no gain) |
| learnable per-layer amp gain alpha (decouples drive from gamma) | 71.2 (no gain, kept: it is exportable via vsat) |
| float (unconstrained) readout, ternary convs | 71.7 (readout is not the bottleneck) |
| warm start from float (60 ep) + 40 ep ternary fine-tune | 71.8 |
| knowledge distillation from the float net (alpha 0.7, T 2) | 70.3 |
| progressive ternarisation (lam ramp over first 50 %) | 71.9; seeds 1/2/3: 74.4 / 71.3 / 70.3 (seed noise ~ +-2 points) |
| width 16 / 32 ternary | 81.0 / 84.3 |

Adopted recipe (deviates from the plan's lr 3e-3 / 20 epochs, which underfits badly): Adam lr 3e-2, cosine to 0, 60 epochs,
batch 256, random IQ-rotation augmentation, progressive ternarisation over the first 50 % of epochs, 3 seeds per model with the
seed chosen by **training-set** accuracy (the test set is never used for selection; train accuracy is a fair selector because
the model underfits: train 73.4 % vs test 73.5 %). The per-line scale fix was tried and is **not** used (`--line-gain` stays
available; `L{l}_gain` is all ones in the shipped model).

## Deviations / notes

- Float baseline stored as schema `dlnn-model-float/1` (see `docs/model_export.md`); `load_model` rejects it on purpose.
  The ternary file is a plain `dlnn-model/1`.
- `gamma` is exported as vsat = 1/(alpha*gamma) (input-referred gain), not as the output gain; see `docs/model_export.md`.
- Readout "ternary linear" is stored as float32 `readout_W = gamma_r*exp(log_ra)*{-1,0,+1}` (digital, contract-compatible).
- Training-time fake-quant is 8-bit on input and layer outputs; reported accuracies are the no-fake-quant eval that the V tier computes
  (with fake-quant: ternary 0.735, same).
- Train/test: `modrec.dataset()` seeds 20260101 / 20260102 (disjoint generations, sha256 `e1b8e744...` in the report).
- Not done: C and R tiers were not exercised with this model.
