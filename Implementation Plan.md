# JEPA vs Decision Model for Agents — Implementation Plan

Written 2026-11-11. Companion to `JEPA vs Decision Model for Agents — Experiment Plan.md`, which explains the idea and the background. This file is the build plan: what to change in the design, what to build in what order, and when to stop.

**Status (2026-11-14):** Phases 0 to 3 are done. Phase 1 collected 270 games (`data/logs/qwen38-main/`) and passed the headroom gate. Phase 3 passed its gate: the rank-aware decision model beats the LLM's first choice by 13.4 points in-distribution. The decision model is ahead of the JEPA arm in-distribution; see "Phase 3 results" below. Phase 4 finished on 2026-11-15: the decision model wins 72.4% of unseen games, JEPA 66.4%, the LLM alone 63.4%; see "Phase 4 results". Phase 5 (lookahead) is optional and not started.

**Machine:** Apple M4, 16 GB RAM, Python 3.13 system-wide, `uv` installed, PyTorch not installed.

---

## 1. The question the experiment answers

Given the same state and the same list of candidate actions, does a selector that first predicts the *outcome embedding* of each action (JEPA world model) choose better actions than a selector that scores (state, action) directly (decision model)?

"Better" means: more useful actions chosen, fewer failed calls, higher task success, and smaller loss of performance on task types and actions it was not trained on.

---

## 2. Design changes from the experiment doc

The experiment doc's design could not separate the JEPA arm from the decision arm as written. These four changes fix that.

### 2.1 The value head takes the task goal as input

The doc has the value head classify the predicted outcome embedding alone. That cannot work: "You open the drawer and see a pen" is useful when the task needs a pen and a dead end otherwise. The value head is therefore `value(predicted_outcome_emb, goal_emb) → {useful, error, dead end}`.

### 2.2 The decision model is a scorer over the same inputs

The doc describes the decision model as state → action. Here it is `score(state, action) → P(useful)` over the same candidate list, with the same frozen encoder and a similar parameter count to the JEPA arm. The only difference between the two arms is then whether the model passes through a predicted outcome embedding.

### 2.3 Every candidate gets labelled, not just the executed one

ALFWorld's text environment is deterministic (confirmed in Phase 0). So for each step, each of the N candidates can be executed by resetting the game and replaying the action prefix. This gives:

- about N times more (state, action, outcome, label) rows per LLM call;
- an offline benchmark where the true outcome of every candidate is known, so selectors can be compared without running the agent again.

### 2.4 Candidates are the LLM's free-text proposals

ALFWorld offers a list of admissible commands at each step. If candidates came from that list, the "error" class ("Nothing happens.") would almost never occur. Candidates are therefore whatever the LLM proposes, valid or not.

---

## 3. Definitions

| Term | Definition |
|---|---|
| **State** | Four separate embeddings: task goal, current observation, last k observations, last k actions. Embedded separately because all-mpnet-base-v2 truncates at 384 tokens. |
| **Action** | The candidate command as text, e.g. `open drawer 2`. |
| **Outcome** | The observation text the environment returns after the action. |
| **Candidate set** | The N=5 ranked actions the LLM proposes at a step. |
| **Steps to go** | The length of the shortest plan from the current state to winning the game. Computed exactly from the game's facts by `src/progress.py`. |
| **Useful** | Steps to go is lower after the action than before, or the action wins the game. |
| **Error** | The outcome is "Nothing happens." |
| **Dead end** | Anything else: a valid action that made no progress. |
| **Oracle** | A selector that picks a useful candidate whenever one exists. The upper bound for any selector. |

**Why steps to go is computed by our own code.** The plan was to read the remaining plan length from TextWorld's built-in planner. That planner is fast in bedrooms and living rooms but takes seconds to minutes per call in kitchens, where the clean, heat and cool tasks live, and its search cannot be tuned. The six ALFWorld goals have a fixed shape, so `src/progress.py` finds the shortest plan directly by trying each choice of object, target, appliance and place to drop a held item. It takes milliseconds.

**Two properties of the label to keep in mind:**

- It uses hidden information. "go to desk 1" is useful only if the target object is on desk 1, which the agent cannot know before looking. Reasonable exploration that finds nothing is labelled a dead end. No selector can predict this label perfectly; all arms face the same limit.
- It follows the game's real win condition, including a quirk: an object can be listed in two receptacles at once (a mug on a coffee machine that stands on a countertop). Taking it "from the countertop" leaves it counted as on the coffee machine.

---

## 4. The four arms

All arms share one LLM call per step, which returns N=5 ranked candidates. They differ only in which candidate is executed.

| Arm | How it chooses |
|---|---|
| **1. LLM-only** | Executes the LLM's first-ranked candidate. |
| **2. Decision model** | Executes the candidate with the highest `score(state, action)`. |
| **3. JEPA world model** | Predicts each candidate's outcome embedding, runs the value head on (predicted outcome, goal), executes the candidate with the highest P(useful). |
| **4. Hybrid** | The world model removes candidates it predicts as error or dead end; the decision model picks among the rest. If none survive, falls back to the decision model over all candidates. |

Reference points reported alongside: random candidate, and the oracle.

### Rank-aware variants (added 2026-11-13)

Arms 2 to 4 as defined above ignore the order in which the proposer ranked its candidates, and that order carries real information. Each therefore also has a `_rank` variant that receives the proposer's rank of the candidate as an extra input at the decision layer: the value head in the JEPA arm, the trunk in the decision arm. These variants re-rank the proposer's list instead of replacing its judgment. The predictor never sees the rank, and both arms get it in the same way, so the comparison between them stays fair. The offline evaluation reports every arm with and without rank.

### Ablations

- **Value head on true outcome embeddings.** Shows how much is lost to prediction error in the predictor, as opposed to the value head.
- **Decision model with an auxiliary outcome-prediction loss.** If this matches the JEPA arm, the gain comes from the extra training signal. If the JEPA arm still wins, the gain comes from routing the decision through the predicted outcome.

---

## 5. Phases

### Phase 0 — Setup and environment checks (done)

Run with `uv run python -m src.phase0_check 10 6` (games per task type, workers). Results on 60 training games, 10 per task type, 1,408 transitions:

| Check | Result |
|---|---|
| Following the computed shortest plan lowers steps to go by one each step and wins, from the start and from states reached by random actions | 60 of 60 games |
| No action lowers steps to go by more than one | 0 violations |
| Start-state steps to go equals the TextWorld planner's plan length | 54 of 54 games where the planner answered within 20 s; 6 timed out |
| Same actions give the same observations on replay | 60 of 60 games |
| Counterfactual branching matches actually taking the action | 0 mismatches |

Setup notes:

- Python 3.11 via `uv`; ALFWorld 0.4.2, TextWorld 1.7.0, PyTorch 2.15.0.
- Game files are in `data/alfworld/json_2.1.1/` (1.6 GB, not committed): 3,553 training games, 140 seen-split and 134 unseen-split validation games.
- `src/tw_speedups.py` caches three TextWorld internals that are recomputed on every step or reset. Traces are identical with and without the caches, and runs are about 4 times faster. Set `ALF_NO_SPEEDUPS=1` to turn them off.
- Speed: 6 to 18 ms per step; 0.3 to 1.2 s to label 5 candidates by replay, depending on room size.

### Phase 1 — Data collection (about 5k steps)

1. Write the agent loop: prompt with goal, history and current observation; ask for 5 ranked candidate actions as JSON.
2. Execute the LLM's top choice, except 20–30% of the time execute a random candidate instead, so the logs cover states a good agent would not reach.
3. For every step, replay to get the outcome and label of all 5 candidates.
4. Log one row per (step, candidate): game id, task type, step index, goal, observation, history, action text, LLM rank, outcome text, label, whether executed.
5. Store as Parquet or JSONL. Split by game, never by step.

**Gate:** compute the headroom, meaning how often the LLM's top choice is not useful while another candidate in the set is. If it is under about 5 percentage points, no selector can show an effect. Use a weaker model, a higher temperature or a larger N, and collect again.

**Expected size:** 5k steps × 5 candidates ≈ 25k labelled rows from roughly 100–170 games.

**Commands:**

```
# Qwen3.8-27B on the AMD endpoint (the chosen proposer). Reasoning is off unless --thinking is given.
uv run python -m src.collect --run NAME --provider openai --model Qwen3.8-27B \
    --base-url https://api.tokenlab.amd.com/v1 --api-key-env AMD_API_KEY --rpm 18 --workers 2

# Claude via the Anthropic SDK (default). Uses the `ant auth login` profile or ANTHROPIC_API_KEY.
uv run python -m src.collect --run NAME --games-per-type 25          # claude-haiku-5-5, thinking off, low effort

# Any OpenAI-compatible endpoint (OpenAI, Ollama, ...)
uv run python -m src.collect --run NAME --provider openai --model MODEL [--base-url http://localhost:11434/v1]

# Stats for a run, including the headroom gate
uv run python -m src.collect --run NAME --summary
```

Logs go to `data/logs/NAME/steps.jsonl`, one JSON object per step with all candidates nested. Re-running the same command resumes. Defaults: 5 candidates, 25% random execution, 40 steps per episode.

**Pilots (12 training games each, Qwen3.8-27B, reasoning off):**

| Measure | Pilot 1 (first prompt) | Pilot 2 (corrected prompt) |
|---|---|---|
| Episodes won | 36% | 83% |
| First choice rejected by the game | 55.7% | 10.3% |
| First choice useful | 9.7% | 29.6% |
| Some candidate useful (oracle) | 17.9% | 55.4% |
| Headroom | 8.2 points | 25.8 points |

Pilot 1's prompt left out rules of the game (no `go to` the current place, `examine` only a held object, a lamp is reached through the furniture it is on) and let the room description scroll out of the history. Pilot 2's prompt states the rules and always includes the opening room description; it adds no hidden information. Pilot 1's data is not used. Both pilots together cost under $0.01.

### Phase 2 — Training on frozen embeddings

1. Embed all goals, observations, actions and outcomes once with all-mpnet-base-v2 and cache to disk.
2. **Predictor (JEPA arm):** MLP, a few million parameters, (state, action) → outcome embedding, cosine loss.
3. **Value head (JEPA arm):** small MLP, (outcome embedding, goal) → 3 classes. Train on true outcome embeddings, then test on predicted ones; also try training on predicted ones.
4. **Decision model:** MLP of matched size, (state, action) → 3 classes.
5. **Hybrid:** no new training; combines 2–4.
6. Train the two ablations from section 4.
7. Five seeds per model.

Training runs on the M4 (MPS or CPU) in minutes; the models are small and the encoder is frozen.

### Phase 3 — Offline evaluation

Rank the candidates of each held-out step with each selector and compare against the known labels.

**Metrics:**

- Top-1 usefulness: share of steps where the chosen candidate is useful.
- Error rate and dead-end rate of the chosen candidate.
- Share of the oracle gap closed, relative to LLM-only.
- Predictor quality: cosine similarity between predicted and true outcome embeddings, split by label.

**Splits:**

- **In-distribution:** held-out games of task types seen in training.
- **Held-out task type:** train on five of ALFWorld's six task types, test on the sixth.
- **Held-out action verb:** remove all rows with one verb (for example `cool`) from training and test on them. This is the test of the doc's "handles new tools better" claim.

**Statistics:** five seeds, bootstrap confidence intervals resampled over games, paired comparisons between arms on the same steps.

**Gate:** go on to Phase 4 only if at least one learned selector beats the LLM's own ranking with a confidence interval that excludes zero. Otherwise the online run cannot show anything the offline run did not.

### Phase 1 results (2026-11-14, `qwen38-main`)

270 training games (45 per task type), 6,157 steps, 30,767 labelled candidates, 0 failed games.

| Measure | Value |
|---|---|
| Episodes won | 70% |
| First choice useful | 25.9% |
| First choice rejected by the game | 17.6% |
| Some candidate useful (oracle) | 47.2% |
| Headroom | 21.3 points (gate: 5) |
| Candidate labels | 71% dead end, 18% error, 11% useful |

### Phase 3 results (2026-11-14, 5 seeds, `results/qwen38-main/offline.json`)

Share of steps where the chosen candidate is useful. Brackets are 95% intervals for the difference from the LLM's first choice.

**In-distribution (42 test games, 860 steps):**

| Selector | Useful | vs first choice | Error |
|---|---|---|---|
| LLM first choice | 28.0% | — | 17.9% |
| Decision | 37.6% | +9.6 [+4.6, +14.6] | 16.8% |
| JEPA | 35.2% | +7.2 [+2.0, +12.2] | 12.1% |
| Hybrid | 37.9% | +9.9 [+5.0, +14.7] | 15.7% |
| Decision, rank-aware | 41.5% | +13.4 [+8.9, +18.3] | 13.0% |
| JEPA, rank-aware | 37.5% | +9.5 [+5.3, +13.9] | 10.3% |
| Hybrid, rank-aware | 40.8% | +12.7 [+8.4, +17.5] | 13.1% |
| Decision + auxiliary loss | 38.4% | +10.3 [+5.4, +15.5] | 15.6% |
| Value head on the true outcome (not deployable) | 42.0% | +13.9 [+9.2, +19.0] | 0.8% |
| Oracle | 55.9% | +27.9 | 0.8% |

- JEPA minus decision: −2.4 [−4.3, −0.5]. Rank-aware: −3.9 [−6.5, −1.7]. The decision model is ahead in both.
- The JEPA arm picks fewer rejected commands than the decision model in every split.
- A value head given the true outcome reaches 42.0%, about the same as the rank-aware decision model. Most of the remaining gap to the oracle is therefore not the predictor's fault: the outcome text alone does not say whether a step made progress.

**Held-out task type (train on five, test on the sixth; 45 test games each):**

| Held-out type | First choice | Decision, rank-aware | JEPA, rank-aware | JEPA − decision (rank-aware) |
|---|---|---|---|---|
| pick_and_place_simple | 29.8% | +4.0 [−0.1, +8.6] | +2.8 [−0.5, +6.5] | −1.2 [−4.5, +1.9] |
| look_at_obj_in_light | 11.7% | +2.0 [+0.2, +3.8] | +2.4 [+1.1, +3.8] | +0.4 [−1.1, +1.8] |
| pick_clean_then_place | 23.3% | +2.5 [−1.9, +6.7] | +3.3 [−0.9, +7.5] | +0.7 [−1.3, +2.7] |
| pick_heat_then_place | 30.5% | +5.9 [+2.0, +9.9] | +0.9 [−3.0, +4.5] | −5.0 [−7.5, −2.9] |
| pick_cool_then_place | 27.2% | +5.2 [+1.7, +8.7] | +1.6 [−1.5, +4.8] | −3.5 [−5.6, −1.9] |
| pick_two_obj_and_place | 35.3% | +5.7 [+1.6, +9.8] | +4.2 [+0.6, +7.9] | −1.5 [−4.1, +1.0] |

Without the rank input, neither model beats the first choice on any held-out task type (all intervals include zero). With it, gains shrink to 2 to 6 points. The world model does not generalise better than the decision model to a new task type.

**Held-out verb:** too few test steps to conclude (20 to 41 steps, 6 to 9 games per verb). Without the rank input both models fall below the first choice on `cool`, `heat` and `use`. On the held-out verb's own rows the JEPA arm separates useful from not useful better than the decision model for `cool` (AUC 0.86 vs 0.63) and `heat` (0.66 vs 0.36); this is the only place the doc's "handles new tools better" claim shows any support, and it does not turn into better choices.

**Gate: passed.** Phase 4 uses the rank-aware arms, because only they hold up outside the training distribution.

### Phase 4 — Online evaluation

```
uv run python -m src.eval_online --train-run qwen38-main --run online-main --provider openai --model Qwen3.8-27B \
    --base-url https://api.tokenlab.amd.com/v1 --api-key-env AMD_API_KEY --rpm 18 --workers 2
uv run python -m src.eval_online --train-run qwen38-main --run online-main --summary
```

Arms: `llm_first`, `decision_rank`, `jepa_rank`, `hybrid_rank`. Selectors are trained once on all 270 games (85 / 15 by game, one seed). Within a game the proposer's answer is cached by prompt, so arms see the same candidates until a selector picks differently. Logs: `data/logs/online-main/online.jsonl`; results: `results/online-main/online.json`. Re-running resumes.

1. Run all four arms on ALFWorld's unseen split (134 games), with the same games for every arm.
2. Record per game: success, steps taken, failed calls (error outcomes), LLM calls, wall-clock time, selector latency.
3. Report success rate with confidence intervals and paired differences between arms.

**Statistical power:** with 134 games the interval on a success rate is about ±8 percentage points, so only large differences will be visible online. The offline results in Phase 3 are the more sensitive test; Phase 4 shows whether offline gains carry over to whole tasks.

**Cost:** roughly 4 arms × 134 games × about 30 steps ≈ 16k LLM calls per run. Default is one LLM seed, varying only the selector seeds.

### Phase 4 results (2026-11-15, `results/online-main/online.json`)

134 unseen games, 0 failed, 7,839 LLM calls, one LLM seed and one selector seed, 40-step limit.

| Arm | Games won | vs LLM first choice | Mean steps | Rejected commands | Useful steps |
|---|---|---|---|---|---|
| LLM first choice | 63.4% | — | 22.8 | 22.9% | 20.7% |
| Decision, rank-aware | 72.4% | +9.0 [−0.7, +18.7] | 18.2 | 16.3% | 31.5% |
| JEPA, rank-aware | 66.4% | +3.0 [−4.5, +11.2] | 20.3 | 16.5% | 28.5% |
| Hybrid, rank-aware | 72.4% | +9.0 [+0.0, +17.9] | 18.3 | 17.9% | 31.2% |

JEPA minus decision: −6.0 [−13.4, +1.5]. Mean steps counts a failed game as 40.

| Task type | Games | LLM first | Decision | JEPA | Hybrid |
|---|---|---|---|---|---|
| pick_and_place_simple | 24 | 88% | 88% | 83% | 88% |
| look_at_obj_in_light | 18 | 61% | 83% | 72% | 83% |
| pick_clean_then_place_in_recep | 31 | 58% | 71% | 65% | 71% |
| pick_heat_then_place_in_recep | 23 | 52% | 74% | 65% | 78% |
| pick_cool_then_place_in_recep | 21 | 67% | 71% | 71% | 67% |
| pick_two_obj_and_place | 17 | 53% | 41% | 35% | 41% |

- The order of the arms matches the offline results: decision ≥ hybrid > JEPA > LLM alone.
- The decision model's gain in games won (+9.0) sits at the edge of significance; the interval just reaches zero. Its per-step gains are clearer: useful steps rise from 20.7% to 31.5% and rejected commands fall from 22.9% to 16.3%.
- JEPA's gain in games won (+3.0) is not distinguishable from zero.
- Every selector does worse than the LLM alone on `pick_two_obj_and_place` (17 games). Not yet investigated.
- The hybrid adds nothing over the decision model.

**Conclusion for single-step action selection:** the direct decision model is at least as good as the JEPA arm in every test, and is simpler. The hypothesis that predicting the outcome first gives better choices is not supported.

### Phase 5 — Lookahead and extras

Only if Phase 3 or 4 shows the world model is competitive.

**2-step lookahead result (2026-11-15, offline, `uv run python -m src.lookahead --run qwen38-main`, `results/qwen38-main/lookahead.json`):** no gain. For each candidate the predictor's outcome is used as the next observation, the other candidates of the step are scored as a second move, and `score = one-step + lam × best second-step`, with `lam` chosen on validation games. In-distribution, 42 test games, 5 seeds:

| Selector | Useful | Change from one-step |
|---|---|---|
| JEPA, one-step | 37.6% | — |
| JEPA + lookahead | 37.7% | +0.1 [−0.4, +0.7] |
| JEPA + lookahead on the true first outcome | 37.6% | 0.0 [−0.7, +0.8] |
| Decision, one-step | 41.5% | — |
| Decision + lookahead | 41.3% | −0.2 [−0.3, 0.0] |
| Decision + lookahead on the true first outcome | 39.8% | −1.6 [−2.6, −0.7] |

Validation picked `lam = 0` (no lookahead) in about half the seeds. Lookahead does not help even with the true first outcome, so the predictor's accuracy is not the limit. JEPA with lookahead is still 3.7 points behind the one-step decision model [−6.2, −1.5]. Not run online. Limits of this test: second-step candidates are only the step's other four proposals, the imagined state keeps the old history, and the one-step label already encodes shortest-path progress, which leaves lookahead little to add.

**Why the selectors fail on `pick_two_obj_and_place` (from the Phase 4 traces):** after taking the first object they go for the second while still holding the first; the robot holds one object, so `take` fails until the step limit. The state input averages the history lines, which loses "I am holding X".

- **2-step lookahead (original note):** feed the predicted outcome embedding back in as the next observation embedding, score second-step actions, and choose the first action by the best two-step value. Open problem: where the second-step candidates come from without another LLM call. Options are reusing the remaining first-step candidates or sampling from command templates.
- **Good views:** pair each raw outcome with a summary from a different model and train the encoder on the pair.
- **SIGReg:** fine-tune the encoder with LeJEPA's regulariser instead of keeping it frozen.
- **Second domain:** repeat on WebShop or the SWE-agent trajectory datasets.

---

## 6. Proposed repository layout

```
pyproject.toml
src/
  env.py          ALFWorld wrapper: reset, step, replay, labels            (built)
  progress.py     Exact steps-to-go from game facts                        (built)
  tw_speedups.py  Caches for slow TextWorld internals                      (built)
  phase0_check.py Phase 0 checks                                           (built)
  agent.py        LLM candidate proposer, plus a stand-in for tests        (built)
  collect.py      Phase 1 data collection and headroom stats               (built)
  embed.py        Frozen sentence embeddings, cached per run               (built)
  models.py       Predictor, value head, decision model, training loop     (built)
  eval_offline.py Phases 2-3: train all selectors, compare offline         (built)
  eval_online.py  Phase 4: whole games with each selector choosing         (built)
data/             Logs and cached embeddings (not committed)
results/          Metrics and plots
```

---

## 7. Open decisions

| Decision | Recommendation | Why |
|---|---|---|
| LLM for candidate proposals | **Decided 2026-11-13:** `Qwen3.8-27B` on AMD's hosted endpoint (`https://api.tokenlab.amd.com/v1`, key in `AMD_API_KEY`), with reasoning turned off. | Chosen by the user. With reasoning on, a call takes 30 to 110 s; with it off, about 3.5 s. Limits on the key: 20 requests per minute and a $10 budget; a call costs under $0.0001. The Claude proposer (`--provider anthropic`) is built but untested, because the saved `ant` login has expired. |
| Number of LLM seeds in Phase 4 | One | Each extra seed is about 16k calls; selector seeds are free. |
| Held-out task type | To be chosen after Phase 1 | Pick one with enough games in the logs to test on. |
| Held-out action verb | `cool` or `heat` | Common enough to test on, and tied to a specific appliance. |

---

## 8. Risks

- **The steps-to-go calculation is wrong in some state Phase 0 did not sample.** It matched the planner and passed both bound checks on 60 games, but it is our own code, not the game's. Re-run `src.phase0_check` with more games if labels look odd.
- **The label uses hidden information** (section 3), which caps how well any selector can do.
- **No headroom.** The LLM's first choice is nearly always as good as any other candidate. Caught by the Phase 1 gate.
- **Blurry predictions.** With a cosine or MSE loss, the predictor averages over possible outcomes when it is unsure. "Nothing happens." is a single fixed vector and easy to predict, so the world model may detect errors well and separate useful from dead end poorly. The per-label predictor metric in Phase 3 shows this.
- **The two arms are the same function class.** Predictor plus value head is still a map from (state, action) to a label. A win for the JEPA arm must be attributed with the ablations in section 4, not assumed to come from "world modelling".

---

## 9. Notes on the experiment doc

- The V-JEPA 2 link has a stray `j` at the end (`2506.09985j`); the ID should be `2506.09985`.
- The other links the doc flags as "from memory" have not been checked.
