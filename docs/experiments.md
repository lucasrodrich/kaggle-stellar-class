# Experiment Campaign: Chasing the Last +0.001 (and Documenting Every Dead End)

> After the [diagnostic investigation](diagnostics.md) showed the pipeline was clean
> and the ~0.011 CV→LB gap was concept drift, I ran a controlled campaign to try to
> beat the baseline anyway. **Nothing beat it.** This is the honest log of what was
> tried, what it scored, and why each was kept or discarded — because a rigorous
> negative result is worth more than a lucky number.

## Ground rules (learned from the diagnostics)

- **Every variant is one change against a fixed baseline** (LightGBM, 27 features) so
  each result is attributable (an *ablation*).
- **Noise floor:** CV differences below ~0.003, and LB differences below ~±0.001, are
  indistinguishable from luck and are **discarded**.
- **CV-gated vs LB-gated:** most variants are judged on CV. The external-data variant
  is judged on the **leaderboard**, because the test set is synthetic and CV
  structurally cannot see the effect (see Tier 2).

## Reference scores

| Submission | CV | Public LB | Private LB |
|---|---|---|---|
| **LightGBM baseline (best)** | 0.96805 | 0.95657 | **0.95736** |
| LightGBM + XGBoost blend | 0.96809 | 0.95644 | 0.95717 |

## Tier 1 — CV-screened variants (`src/experiments.py`, `src/train_fast.py`)

| Variant | CV | Δ vs baseline | Decision |
|---|---|---|---|
| baseline (27 features) | 0.96805 | — | reference |
| + 8 engineered features (curvatures, interactions, `low_z` flag) | 0.96797 | −0.00008 | **discard** |
| + out-of-fold target encoding | 0.96805 | +0.00000 | **discard** |
| drop `alpha` / `delta` (sky position) | 0.95422 | **−0.01383** | **discard** |
| target encoding + drop position | 0.95421 | −0.01384 | **discard** |
| tuned (num_leaves 255, lr 0.02, λ 5) | 0.96806 | +0.00001 | **discard** |

**Nothing cleared +0.003.** Two findings worth keeping:

- **More engineered features didn't help** (they even hurt slightly). They were
  deterministic recombinations of existing columns — and gradient-boosted trees can
  already represent those internally, so pre-computing them adds nothing. (For *tree*
  models, hand-made interactions of existing inputs rarely help — unlike linear models.)
- **Sky position is critical, not noise.** Dropping `alpha`/`delta` cost **0.014** — a
  huge amount — even though the gain-based importance chart ranked them *low*. This is
  **"importance ≠ necessity":** a feature can rank modestly yet capture signal nothing
  else substitutes for, so only *ablation* reveals what is safe to remove. (Position
  likely encodes SDSS survey-selection structure, which correlates with class.)

## Tier 2 — external real data (`src/train_external.py`)

**Hypothesis:** the competition data is synthetic (a generator's imperfect copy of a
real dataset). Mixing in the original **real SDSS17** rows should anchor the decision
boundary to the true label rule the synthetic test approximates — bridging the concept
drift. 100,000 real rows were cleaned (SDSS `-9999` sentinels → NaN; the two
synthetic-only categoricals marked missing) and added to every training fold.

**This is leaderboard-gated:** the test set is synthetic, so real data can *lower*
synthetic-CV while *raising* the LB. CV cannot judge it — only a submission can.

| | CV (synthetic-val) | Public LB | Private LB |
|---|---|---|---|
| baseline | 0.96805 | 0.95657 | 0.95736 |
| + 100k real SDSS rows | 0.96812 | 0.95663 | 0.95732 |
| Δ | +0.00007 | **+0.00006** | **−0.00004** |

**Null result.** The LB moved by +0.00006 (public) / −0.00004 (private) — both inside
the ±0.001 noise floor, and **opposite in sign**, the classic signature of pure noise.
Only 0.37% of predictions changed. **Discard.** The concept drift here is not the kind
of correctable generator label-noise that real data fixes.

## Tier 3 — nearest-neighbour label lookup (`src/knn_lookup.py`)

**Hypothesis:** if the synthetic rows are near-copies of real SDSS objects, the nearest
real neighbour's label recovers the truth better than the model in some regions — the
suspected route to the 0.970 leader cluster.

**Tested for free, offline, on the labelled training set** (no submission spent): for
each train row, find its nearest real neighbour (standardised features) and compare a
"use the real label when the match is very close, else trust the model" hybrid against
the model alone.

| | Accuracy on train |
|---|---|
| Model (baseline OOF) | **0.96805** |
| kNN label (all rows) | 0.88525 |
| Best hybrid (override closest 1%) | 0.96751 |

**Falsified.** Every override threshold made it *worse*. Two reasons: (1) no
near-duplicates exist — the closest synthetic→real distance is 0.029 (median 0.31 in
standardised space), so the generator produced genuinely new points rather than
memorising real rows; (2) 1-NN on 8 raw features is a weak classifier (0.885) next to
the full model. **Discarded without a submission** — the offline test settled it.

## Tier 4 — diverse stacking + local TabPFN (`src/stack.py`, `src/stack_tabpfn.py`)

The leaders' actual technique: diverse base models combined by a logistic-regression
meta-learner (Chris Deotte's "stacker"). Two stacks were built:

- **4-model:** LightGBM + XGBoost + LogisticRegression + MLP.
- **5-model:** the above + **TabPFN**, a tabular foundation model, run **fully locally**
  (weights from a *public* HuggingFace repo, CPU inference, telemetry/browser-login
  disabled — no data leaves the machine).

| | Individual acc | Stack result |
|---|---|---|
| LightGBM / XGBoost | 0.968 / 0.968 | strong |
| LogReg / MLP | 0.932 / 0.958 | weak, diverse |
| TabPFN (CPU, 5k context) | 0.952 | weak (starved of data) |
| 4-model stack (full-train CV) | | 0.96769 (below baseline) |
| **5-model stack + TabPFN (LB)** | | **public 0.95440 / private 0.95522** |

**Both stacks underperformed the baseline** (0.95736 private); the TabPFN stack lost by
~0.002 on the leaderboard. Two reasons:

1. **Diversity from *weak* members doesn't help.** A meta-learner cannot manufacture
   accuracy from models individually far weaker than LightGBM (0.93–0.96 vs 0.968); it
   only dilutes the strong one.
2. **The key ingredient needs a GPU.** TabPFN is what powers the leaders' stacks, but its
   strength requires feeding it lots of data at GPU speed. On CPU (~86 rows/s) only a
   5k-row context of 577k was feasible, crippling exactly the member that mattered.

A final CV lesson: the 5-model *subsample* CV read 0.9699 while the leaderboard returned
0.955 — a **0.015 gap**. Subsample CV on this competition is worthless; only the LB
judges. This is a hardware ceiling, honestly hit: the method was reproduced, the compute
was not.

**GPU re-run (T4, 16k context).** To rule out the CPU handicap, the stack was rebuilt on
a Kaggle T4 GPU where TabPFN could use a 16k-row context instead of 5k. TabPFN alone
improved (0.952 → **0.9565**), confirming more context helps — but the full stack still
scored **public 0.95410 / private 0.95483**, *below* the baseline and even a hair below
the CPU stack. So even the technique done "right" on a GPU loses here: 16k of 577k rows
(~3%) still isn't enough context for TabPFN to beat the GBMs, and a free T4's 15GB caps
how far the context can grow. The leaders' larger contexts / more ensemble members need
compute beyond free-tier hardware. The runnable GPU notebook is `notebooks/gpu_stack.ipynb`.

## Conclusion

Across engineered features, target encoding, feature pruning, hyperparameter tuning,
model blending, real-data augmentation, nearest-neighbour label lookup, and diverse
stacking with a local TabPFN, **nothing beat the plain LightGBM baseline** (0.95736
private). That is the honest result: on this competition, with this
approach, ~0.957 is the ceiling, and the ~0.011 gap to a hypothetical 0.968 is concept
drift that none of these levers could close.

**What this campaign demonstrates** is not a winning score — it's the *method*:
controlled one-variable experiments, a noise-floor discipline that refuses to chase
luck, a principled CV-vs-LB distinction, and the willingness to run the highest-upside
idea (external data) and **report that it failed**. Knowing what *doesn't* work, and
why, is the harder and more valuable half of the job.

### What the leaders actually did — and where my conclusion was too strong

I originally concluded "~0.957 is the ceiling; blending doesn't help; concept drift caps
it." **Reading the top public notebooks proved that partly wrong**, and the honest
correction matters more than the original claim.

The most-upvoted solutions (Chris Deotte's *GPU Logistic Regression Stacker*, Philipp
Singer's *TabPFN-3 Stacker*, and many *ensemble* / *ridge-flip* / *probability
consensus* notebooks — both authors are Kaggle Grandmasters) reveal that the ~0.97
cluster comes from three things this project did **not** do:

1. **Real model diversity.** Our blend was LightGBM + XGBoost — both gradient-boosted
   trees, ~99.5% correlated, so averaging them did nothing. The leaders add
   *fundamentally different* learners, notably **TabPFN** (a transformer-based tabular
   foundation model) that errs differently from trees. Diversity is what makes a blend
   work; we didn't have it.
2. **Stacking, not simple averaging.** They train a **meta-model** (logistic regression)
   on the out-of-fold predictions of many base models, learning the optimal per-class
   combination — far stronger than a fixed 0.7/0.3 weight.
3. **Post-processing** ("ridge-flip / probability consensus") that flips borderline
   predictions by model agreement.

So two of my earlier statements need correcting:

- *"Blending doesn't help"* → should be *"blending near-identical GBMs doesn't help."*
  Diverse stacking clearly does.
- *"0.957 is the ceiling"* → wrong. The leaders reach 0.973, which is **above even our
  optimistic CV of 0.968**, so there is real signal we left on the table. Part of our
  CV→LB gap was genuine concept drift; part was a single deep GBM **overfitting
  synthetic-specific noise** that a diverse, regularised stack avoids.

The concept-drift *diagnosis* stands (it is real and measured), but calling it an
immovable ceiling was an overreach. Fittingly, the project's own lesson — *never trust a
result you haven't stress-tested* — caught my **own** conclusion once we looked at the
leaderboard. The stacking attempt to close this gap is logged in Tier 4 above: it
reproduced the leaders' method but underperformed, because on CPU the one strong diverse
member (TabPFN) could not be given enough data. The leaders' edge is real — and it is a
GPU-plus-full-data edge, not a trick.
