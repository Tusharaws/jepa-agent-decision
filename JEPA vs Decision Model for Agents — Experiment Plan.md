# JEPA vs Decision Model for Agents — Experiment Plan

## 1. The idea in one line

Train a small JEPA-style world model that predicts the *outcome embedding* of an agent's tool call before running it, then use it to pick actions cheaply. Compare it against a direct decision model trained on the same data.

## 2. Background: what the latest LeCun paper actually shows

Paper: "Proving and Improving the Platonic Representation Hypothesis with JEPAs" (Balestriero & LeCun, arXiv 2610.39302, Oct 29 2026).

- **Main result:** Encoders trained separately on different modalities end up with the same representation, up to a rotation, under four assumptions (finite world, lossless observation, matching positive pairs, whitening anti-collapse).
- **Key practical lesson:** "Good views" (two different observations of one scene) beat "bad views" (two corruptions of one observation). Next-token prediction is formally a bad-view JEPA.
- **Headline claim:** A 109M text JEPA scored 0.184 on WIT alignment vs 0.154 for Qwen2.5-72B. This is representation alignment only, not reasoning, coding or generation.
- **Caveats in the paper itself:** a 4th seed scored 0.121; comparisons with LLMs at the 33M scale are within noise; 2 of 3 training captions came from an image-seeing captioner (removing them drops alignment from 0.107 to 0.060).
- **Important for us:** world-model JEPAs (a learned predictor conditioned on an action) are explicitly NOT covered by the theory.

## 3. JEPA world model vs decision model

- **Decision model (model-free):** state → action. Learns which choices work, not why. Simple and fast for narrow, stable tasks.
- **JEPA world model (model-based):** predicts what each action *would* do before running it, then picks the action with the best predicted result. Flow: (state, action) → predicted outcome embedding → score → choose. See the worked example below.
- **Hybrid:** the world model scores candidates and the decision model picks among them.

### Worked example: how the world model chooses

Task: "find the cheapest red shoes on this site." The LLM suggests three candidate actions: **A.** search "red shoes", **B.** click "Sort by price", **C.** call the checkout API.

1. **Predict the outcome.** For each action, the world model takes the current situation plus the action and predicts what the result will look like. It doesn't write the result out as text. It produces an embedding, a list of numbers that captures the gist, such as "a page of product listings" or "an error message".
2. **Score it.** A small scorer rates each predicted outcome. A looks useful, because it predicts a list of shoes. B looks like a dead end, because there's nothing to sort yet. C looks like an error, because the cart is empty.
3. **Choose.** Only A is run. B and C were never executed, so two wasted calls are saved.

Why this predict-first design helps:

- **Handles new tools better.** A decision model only memorises "in situations like this, pick that", so a new tool leaves it clueless. A world model has learned how actions change things in general, so it can still make a sensible guess about a tool it has rarely seen.
- **Can think a few steps ahead.** Because it predicts the next state, you can feed that prediction back in ("if I search, then sort, what happens?") and plan a short sequence without running anything.
- **You can see why.** When it rejects C, you can inspect the predicted outcome and see that it looked like an error. A decision model just says "not C" with no reason attached.

## 4. Architecture

1. **Encoders:** a frozen sentence-embedding model (e.g. all-mpnet-base-v2) for states, actions and outcomes.
2. **Predictor:** a small MLP or tiny transformer (a few million params) mapping \[state\_emb, action\_emb\] → outcome\_emb. Loss: cosine or MSE in embedding space.
3. **Value head:** a small classifier on the predicted outcome embedding with labels {useful, error, dead end}.
4. **Optional later:** fine-tune the encoder with LeJEPA's SIGReg regulariser to prevent collapse.

**Run-time loop:** the LLM proposes N candidate actions → the predictor and value head score them in milliseconds → execute only the top 1–2 → optionally roll forward 2–3 steps using predicted outcomes.

## 5. Dataset

- **Option A, your own agent logs:** log (state, action, outcome, label) per step. Start with a few thousand steps.
- **Option B, public environments:** ALFWorld (recommended start: small, cheap, easy to score), WebShop, ToolBench.
- **Tip:** include deliberate failures and bad actions so the predictor learns what mistakes look like.
- **Good-views angle (from the paper):** where possible, pair different observations of the same outcome (e.g. the raw tool output plus a summary from a different model) rather than augmentations of one text.

### Do these environments include negative actions?

Mostly no. They are environments, not datasets of mistakes. The training data that comes with them is almost entirely successful runs, so the negatives have to be generated.

- **ALFWorld:** the bundled demos are expert trajectories from a planner, so they are all correct. But the environment reports failed actions ("Nothing happens.") and incomplete tasks, so bad actions are easy to create and label.
- **WebShop:** the human demonstrations are mostly good runs. It gives a graded reward from 0 to 1 for how well the bought item matches the request, which yields "partly wrong" outcomes as well as clear wins and losses.
- **ToolBench:** has some failure signal built in. Its data came from a tree search that explored and abandoned bad branches, and the logs include API errors and invalid calls. The released training set still leans towards successful paths.

**How to get negatives:** run an LLM agent in the environment and inject noise. At some steps, take a random valid action instead of the agent's choice, or use a weaker model. That gives a natural mix of useful, error and dead-end outcomes, all labelled by the environment for free.

**Reference:** ETO (Exploration-based Trajectory Optimization) collected failed trajectories on WebShop and ALFWorld and trained on contrasting good and bad runs. See the reading list.

### Public datasets with real failures (coding agents)

These give large numbers of failed runs without having to generate them.

- **Nebius SWE-agent trajectories:** 80,036 trajectories of a SWE-agent-based agent attempting GitHub issues from SWE-bench-extra and the SWE-bench dev split. 13,389 resolved the issue and 66,647 did not, so it is mostly failures. Each trajectory includes the steps, the exit status, the generated patch and test logs. CC BY 4.0. Load with `load_dataset('nebius/SWE-agent-trajectories')`. [Blog](https://nebius.com/blog/posts/scaling-data-collection-for-training-swe-agents) · [Hugging Face](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)
- **NVIDIA Open-SWE-Traces:** 207,489 trajectories collected with SWE-agent and OpenHands. Each has a `resolved` field: 1 if solved, 0 if not, and -1 if unknown. [Hugging Face](https://huggingface.co/datasets/nvidia/Open-SWE-Traces)

**Caveat:** labels are for the whole trajectory, not each step. To get step-level outcomes (useful, error or dead end), derive them from the tool outputs, for example error messages, failed test runs or repeated actions, or use an LLM to label a sample.

## 6. Experiment design

- **Arms:** (1) LLM-only baseline, (2) decision model, (3) JEPA world model, (4) hybrid.
- **Same data and the same candidate-action generator for every arm.**
- **Metrics:** task success rate, wasted/failed tool calls, LLM calls per task, cost and latency per task, performance on held-out tasks or unseen tools (generalisation).
- **Rigour:** multiple seeds, confidence intervals, a held-out split by task type.

## 7. First milestones

1. Run a baseline LLM agent on ALFWorld and log about 5k steps, including failures.
2. Train the predictor and value head on frozen embeddings.
3. Train the decision model on the same logs.
4. Compare all four arms on a held-out set.
5. Try 2-step lookahead and unseen-task generalisation.

## 8. Reading list

**Core JEPA**

- LeCun, "A Path Towards Autonomous Machine Intelligence" (2022) — https://openreview.net/pdf?id=BZ5a1r-kVsf
- Assran et al., I-JEPA: "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" — https://arxiv.org/abs/2301.08243
- Balestriero & LeCun, "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics" (SIGReg) — https://arxiv.org/abs/2511.08544
- Balestriero & LeCun, "Proving and Improving the Platonic Representation Hypothesis with JEPAs" — https://arxiv.org/abs/2610.39302
- Explainer with appendix caveats (Pebblous) — https://blog.pebblous.ai/report/lecun-platonic-representation-good-views/en/

**JEPA world models and planning (most relevant)**

- Destrade et al., "Value-guided action planning with JEPA world models" — https://arxiv.org/abs/2601.00844
- "LeWorldModel: Stable End-to-End JEPA from Pixels" — https://arxiv.org/abs/2603.19312
- Meta, V-JEPA 2 (video world model for robot planning) — https://arxiv.org/abs/2506.09985j

**JEPA for language**

- Huang, LeCun & Balestriero, "LLM-JEPA" — https://arxiv.org/abs/2509.14252

**Views and representation similarity**

- Tian et al., "What Makes for Good Views for Contrastive Learning?" (InfoMin, 2020) — https://arxiv.org/abs/2005.10243
- Gröger, Wen & Brbić, "Revisiting the Platonic Representation Hypothesis: An Aristotelian View" — https://arxiv.org/abs/2602.14486

**Agent environments**

- ALFWorld — https://arxiv.org/abs/2010.03768
- WebShop — https://arxiv.org/abs/2207.01206
- ToolLLM / ToolBench — https://arxiv.org/abs/2307.16789
- Song et al., "Trial and Error: Exploration-Based Trajectory Optimization for LLM Agents" (ETO), uses failed trajectories on WebShop and ALFWorld (link from memory, so verify) — https://arxiv.org/abs/2403.02502

*Note: double-check the V-JEPA 2, InfoMin, ALFWorld, WebShop and ToolBench links. They were added from memory, not from this session's search.*
