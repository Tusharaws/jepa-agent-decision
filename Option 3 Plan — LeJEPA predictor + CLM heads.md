# Option 3 Plan — LeJEPA predictor + CLM-style projection heads

Written 2026-11-17. Follows `Implementation Plan.md` (Phases 0–5, all done).

**Status (2026-11-17):** started on **AgentProcessBench** instead of ALFWorld or AED, at the user's choice (see section 2.3). Encoder switched from Q4_K_M to **Q8_0**; Q4 removed from Ollama. Code: `src/apb.py` (data), `src/option3.py` (encode, train, compare), `src/clm_eval.py` (CLM encoder and heads). First run finished 2026-11-18; results in section 2.4.

## 2.3 AgentProcessBench (chosen 2026-11-17)

- **Source:** Fan et al. 2026, arXiv 2603.14465; MIT licence; on Hugging Face as `LulaCola/AgentProcessBench`. Downloaded to `data/agentprocessbench/`.
- **Contents:** 1,000 runs, 250 each from BFCL, GAIA, HotPotQA and τ²-Bench. Each source has 50 tasks, each attempted by 5 different LLM agents.
- **Labels:** every assistant step is human-labelled +1 correct, 0 neutral or −1 incorrect (89.1% agreement).
- **What it has:** every tool call is followed by the tool's real response, which is what the predictor learns from.
- **What it lacks:** one action per step, so there are no alternatives at the same point.
  - The test is therefore **telling correct from incorrect steps (AUC)**, not choosing among candidates.
  - First calls are almost all labelled correct, so "best of 5 agents' first move" has little to measure.
- **Steps kept:** the 5,064 tool-call steps (labels: 1,525 incorrect, 377 neutral, 3,162 correct). Neutral steps are used to train the predictor but are left out of the AUC.
- **Situation text:** built only from what the agent saw: the first user message plus the most recent conversation, cut to 5,000 characters. τ²-Bench's `question` field holds the simulated user's hidden instructions and is not used.
- **Label caveat:** once a step is marked incorrect, later steps that follow from it stay incorrect until the agent recovers. A label therefore partly means "the run is off track", not only "this call was bad".
- **Splits:** by task, so all 5 agents' runs of a task stay together.
  - In-distribution: 70 / 15 / 15 within each source.
  - Leave-one-source-out: train on three sources, test on the fourth. This is the unseen-tools test.
- **Selectors:** the gates of section 3 apply with AUC in place of useful-pick rate.

| Selector | What it reads |
|---|---|
| `clm_action`, `clm_true` | CLM's released heads, no training, on the call / on "call → real result" |
| `heads_action` | CLM heads fine-tuned on the call (decision model) |
| `heads_true` | CLM heads fine-tuned on "call → real result" (ceiling) |
| `heads_pred` | CLM heads fine-tuned on the predicted result (**Option 3**) |
| `heads_pred_sigreg` | same, predictor trained with LeJEPA's SIGReg term |
| `mlp_action`, `mlp_true` | plain MLP on the same embeddings, for reference |

- **Seeds and intervals:** 3 seeds; 95% bootstrap intervals over tasks.
- **Main metric:** AUC averaged over the four sources, so differences in error rate between sources do not inflate it.

### 2.4 Results (run finished 2026-11-18; `results/apb/option3.json`)

AUC for telling correct from incorrect steps; 3 seeds; 95% intervals over tasks.

| Selector | In-distribution | Held-out bfcl | Held-out gaia | Held-out hotpotqa | Held-out tau2 |
|---|---|---|---|---|---|
| clm_action (zero-shot) | 0.527 | 0.630 | 0.450 | 0.608 | 0.508 |
| clm_true (zero-shot) | 0.380 | 0.600 | 0.393 | 0.386 | 0.420 |
| heads_action | 0.858 | 0.618 | 0.648 | 0.832 | 0.489 |
| heads_true | 0.874 | 0.603 | 0.695 | 0.629 | 0.666 |
| **heads_pred** | **0.877** | 0.556 | 0.648 | 0.844 | **0.662** |
| heads_pred_sigreg | 0.623 | 0.600 | 0.672 | 0.629 | 0.442 |
| mlp_action | 0.893 | 0.548 | 0.865 | 0.865 | 0.724 |
| mlp_true | 0.904 | 0.653 | 0.865 | 0.840 | 0.855 |

heads_pred − heads_action: in-distribution +0.019 [−0.008, +0.044]; bfcl −0.062 [−0.169, −0.007]; gaia +0.000; hotpotqa +0.012; **tau2 +0.173 [+0.119, +0.230]**.

- **Gate 1 (heads_true > heads_action):** not met in-distribution (+0.015 [−0.029, +0.057]). It is met on held-out tau2 (+0.177) and is mixed elsewhere.
- **Gate 2 (heads_pred ≥ heads_action):** met in-distribution. It is met on gaia, hotpotqa and tau2 and fails on bfcl. On tau2 the predicted result recovers the full gain of the real result.
- **SIGReg (λ = 0.05) hurt:** its predictions have cosine 0.52 to the real result, against 0.97 without it. λ was far too large for a frozen-encoder setup.
- **Zero-shot CLM is near chance.** clm_true is below chance, so CLM's released heads do not judge results as "good".
- **A plain MLP beats CLM's heads** on most splits. mlp_pred was not run, which leaves the fairest Option 3 comparison open.

### 2.5 Follow-up: MLP on the predicted result, small SIGReg (2026-11-18; `results/apb/option3_followup.json`)

| Split | mlp_action | mlp_pred | mlp_true | mlp_pred − mlp_action |
|---|---|---|---|---|
| In-distribution | 0.893 | 0.884 | 0.904 | −0.009 [−0.033, +0.017] |
| Held-out bfcl | 0.548 | 0.552 | 0.653 | +0.004 [−0.029, +0.026] |
| Held-out gaia | 0.865 | 0.839 | 0.865 | −0.025 [−0.039, −0.014] |
| Held-out hotpotqa | 0.865 | 0.861 | 0.840 | −0.005 [−0.042, +0.032] |
| **Held-out tau2** | 0.724 | **0.827** | 0.855 | **+0.103 [+0.070, +0.130]** |

- **tau2:** Option 3 beats the best decision model there and closes about 80% of the gap to the real result.
- **Elsewhere:** the real result adds little (in-distribution, gaia and hotpotqa), so prediction has nothing to add. On gaia it costs a little (−0.025).
- **bfcl:** the real result helps (+0.10), but the predictor captures none of it.
- **SIGReg 0.001 / 0.005:** no consistent effect. The changes are about ±0.02 either way, except that 0.005 hurts the heads on tau2 (−0.117). Cosine to the real result is 0.97 / 0.95, against 0.97 without SIGReg.
- **Conclusion:** predicting the result helps only where the tool's response carries information the call text does not and the predictor can learn it (tau2's stateful customer-service tools). It is not a general win over a decision model. SIGReg adds nothing with a frozen encoder.

---

## 1. The idea

Judge a tool call **before** running it, by predicting what it will return and scoring that prediction against the situation.

```
situation ──► encoder ──────────────────────────────► state head ──┐
                                                                    ├─► score = scale · cos
situation + action ──► encoder ──► LeJEPA predictor ──► action head ┘
                                   (predicted "action → result" embedding)
```

| Part | What it is | Trained on |
|---|---|---|
| Encoder | Qwen3-8B, last-token pooling, frozen (CLM's encoder) | nothing; frozen |
| Predictor | MLP: (situation, action) → embedding of "action → result" | real results of executed actions; **no labels needed** |
| State head, action head | CLM's architecture (4096 → 1536 → 1536 → 512), started from CLM's weights | "useful" labels, contrastive loss |

The predictor's output lands in the encoder's own space, which is what the action head reads. That is how the two connect.

**Why this differs from the JEPA arm that lost earlier:**

- The judge is CLM's pretrained ranker, not a value head trained from scratch.
- The judge sees the full situation; the old value head saw only the goal.
- The encoder is an 8B model rather than a 110M sentence encoder.

---

## 2. Data

### 2.1 AED is not available yet

AED (arXiv 2609.40111, Zhu et al., 2026) has no public download. The paper says the authors "plan to open-source the code on GitHub and release model checkpoints and data" after acceptance, subject to licence and privacy review.

What the paper says each record holds:

- a failed action–observation trace;
- the erroneous step index, a free-text diagnosis, and a proposed correction;
- for 3,062 cases, a replay of both the correction and the original action from the same checkpoint.

Points it does not confirm:

- whether every step stores the tool's response;
- whether the replays store the result text or only pass/fail.

The paper text is CC BY-NC-SA 4.0, so a data release under the same terms would rule out commercial use.

### 2.2 So the plan starts on our ALFWorld data

`data/logs/qwen38-main` already has what Option 3 needs, and in a cleaner form than AED will:

| Need | ALFWorld (have now) | AED (when released) |
|---|---|---|
| Real result of each action | yes, for **all 5 candidates** at every step | executed actions only; corrections mostly not run |
| Label per action | exact: useful / error / dead end | LLM diagnosis of one step per trace (85.5% agreement on a 69-case check) |
| Several actions from the same point | 5 per step, 6,157 steps | 3,062 replay pairs |
| Variety of tools | 1 environment, about 10 verbs | 33 environments, 19 harnesses |

ALFWorld answers "does Option 3 work at all"; AED answers "does it carry over to new tools". ALFWorld comes first.

---

## 3. Steps

### Step 1 — Encode the data with CLM's encoder (no training)

- Serve the embeddings with llama.cpp (`llama-server --embedding --pooling last`), as already set up in `src/clm_eval.py`.
- Embed for every step:
  - the situation text: task, starting room, history, observation, question;
  - each candidate as "action";
  - each candidate as "action → real result".
- **Quantisation:** use **Q8_0** if possible. The model card rates Q4_K_M "not recommended for ranking" (−6.6 points). A partial Q8 download is already in the Ollama folder.
- **Time:** about 3 s per long situation on the M4 with Q4.
  - 6,157 situations take about 5 hours.
  - The roughly 31k short candidate texts take about 2 hours.
  - The run is resumable and cached to disk.

### Step 2 — Zero-shot baselines (no training)

Using the cached embeddings and CLM's own heads, on the 42-game test split from before:

- **A.** CLM ranks the actions.
- **B.** CLM ranks "action → real result". This is the ceiling: a perfect predictor feeding untrained heads.

These are what `src/clm_eval.py` was about to measure when it was stopped.

### Step 3 — Train the predictor (LeJEPA)

- **Input:** encoder embeddings of the situation and the action.
- **Target:** encoder embedding of "action → real result", for every candidate (all 5 were executed by replay).
- **Loss:** cosine loss, plus LeJEPA's SIGReg term on the predicted embeddings to keep them spread out rather than averaged into a blur.
- **On the term "LeJEPA":** full LeJEPA also trains the encoder. With a frozen 8B encoder on a 16 GB Mac, only the predictor is trained, and the targets are fixed, so collapse cannot happen. SIGReg acts as a regulariser here, not a collapse guard.
- **Optional:** add a small trainable adapter on top of the frozen encoder with SIGReg, which is the affordable version of training the encoder. Do this only if Step 5 shows the predictor is the bottleneck.

### Step 4 — Fine-tune the heads

- **Start:** CLM's released heads.
- **Loss:** contrastive (InfoNCE), as CLM was trained.
  - **Positive:** the step's useful candidate.
  - **Hard negatives:** the error and dead-end candidates of the **same step**.
  - **Other negatives:** candidates from other steps in the batch.
- **Three versions**, same size and same data:

| Version | Action head reads | Role |
|---|---|---|
| H-action | the action text only | decision model (the comparator) |
| H-true | "action → real result" | ceiling, not deployable |
| **H-pred (Option 3)** | the predictor's output | the idea being tested |

- H-pred is trained on the predictor's own outputs, as the earlier JEPA value head was, since that is what it sees in use.
- Each version is also run with the LLM's rank as an input, since the rank-aware versions did best before.

### Step 5 — Offline comparison

- **Split and seeds:** the same splits and 5 seeds as `src/eval_offline.py`.
- **Splits:**
  - in-distribution (42 test games);
  - held-out task type (all six in turn);
  - held-out verb (`cool`, `heat`, `clean`, `use`).
- **Metrics:**
  - useful-pick rate;
  - error rate;
  - gap closed relative to the oracle;
  - paired differences between versions, with bootstrap intervals over games.
- **Reference points from before:**

| Earlier result (test split) | Useful picks |
|---|---|
| LLM first choice | 28.0% |
| JEPA arm, rank-aware | 37.5% |
| Decision MLP, rank-aware | 41.5% |
| Value head on real result | 42.0% |
| Oracle | 55.9% |

**Gates:**

1. **H-true must beat H-action.** If even the real result does not help the heads, no predictor can make Option 3 win. That would repeat the earlier finding, and we stop.
2. **H-pred must reach at least H-action**, with the interval of the difference not entirely below zero.
3. **Run online only if gate 2 passes.** That means 134 unseen games, about 9 hours of Qwen calls, re-using `src/eval_online.py`.

### Step 6 — AED, when it is released

1. Check the schema:
   - per-step tool responses present?
   - replay result text present?
   - licence terms?
2. Build training pairs:
   - **positive:** the correction, using its replay result where stored;
   - **hard negative:** the faulty action with its real result;
   - **predictor targets:** only the actions that were actually executed.
3. Hold out whole environments (for example 6 of 33) to test unseen tools.
4. Metrics:
   - how often each version prefers the correction over the faulty action;
   - whether the lowest-scoring step matches AED's labelled error step.

**If AED stays unreleased:**

- AgentErrorBench (AgentDebug, 200 annotated failures, 3 environments) is small but public.
- Or generate AED-style data ourselves in a second environment with the same replay method as ALFWorld. ScienceWorld and TextWorld Cooking are both in AED's list.

---

## 4. Code to add

```
src/
  clm_encode.py     Step 1: embed everything with the Qwen3-8B encoder, cached
  clm_eval.py       Step 2: zero-shot baselines (exists)
  option3.py        Steps 3-5: predictor, head fine-tuning, offline comparison
  eval_online.py    re-used for the online run, with an Option 3 selector added
```

No new LLM API cost until the online run.

---

## 5. Risks

- **The heads may not need the result** (gate 1). Our earlier finding was that knowing the result helped no more than judging the action directly. This is the most likely way Option 3 fails, and gate 1 tests it cheaply.
- **Speed:** an 8B encoder on a 16 GB Mac is slow (Step 1 takes about 7 hours) and leaves little memory for anything else. Q8 is more accurate but larger (8.7 GB).
- **Label noise on AED:** diagnoses are LLM-written; the paper's own check is 85.5% step agreement on 69 cases.
- **AED may arrive late, or without per-step responses.** The ALFWorld path stands on its own either way.

---

## 6. Decisions needed before starting

| Decision | Recommendation |
|---|---|
| Encoder quantisation | Finish the Q8_0 download; use Q4 only for a quick trial run |
| Run Step 1 on all 6,157 steps or a subset first | Test split plus 60 training games first (about 1.5 hours), to pass gate 1 cheaply before encoding everything |
| Train an adapter on the encoder (full LeJEPA-style) | Only if Step 5 shows the predictor is the bottleneck |

---

## Sources

- AED paper: https://arxiv.org/abs/2609.40111 (full text: https://arxiv.org/html/2609.40111v2)
- CLM heads: https://huggingface.co/Contrastive-LM/CLM-v0.1-8B
- CLM GGUF and quantisation table: https://huggingface.co/czl/CLM-v0.1-8B-GGUF
- AgentDebug / AgentErrorBench: https://github.com/ulab-uiuc/AgentDebug
