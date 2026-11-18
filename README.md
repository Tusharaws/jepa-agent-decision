# JEPA world model vs decision model for agent tool calls

When an agent chooses its next tool call, does it help to **predict what the tool will return** (a JEPA-style world model) and judge that prediction, rather than judging the call directly (a decision model)?

This repo tests that question in two settings:

1. **ALFWorld**, a text household game. An LLM proposes 5 candidate actions per step, and a learned selector picks one.
2. **AgentProcessBench**, real agent runs with human per-step labels. Here a judge scores whether a tool call is correct, given the call, its real result, or a predicted result.

## Key finding: predicting the result helps on stateful tools

On **tau2**, whose stateful customer-service tools (bookings, orders, accounts) were never seen in training, the LeJEPA predictor + CLM-style projection heads clearly beat judging the call alone:

| Judge, tested on held-out tau2 | Judges the call only | Judges the **predicted** result (LeJEPA predictor) | Judges the real result (ceiling) |
|---|---|---|---|
| CLM-style projection heads | 0.489 | **0.662** (+0.173 [+0.119, +0.230]) | 0.666 |
| Plain MLP | 0.724 | **0.827** (+0.103 [+0.070, +0.130]) | 0.855 |

AUC of correct vs incorrect steps; 3 seeds; 95% bootstrap intervals over tasks.

- **Why it helps on stateful tools:** the outcome of a stateful call depends on hidden state, such as whether that order exists or the booking can be changed, which the call text alone doesn't show.
- **How much it recovers:** the predictor learns this from other tools. With CLM's heads it recovers essentially all of what the real result is worth; with the plain MLP it recovers about 80%.
- **On the lookup-style sources** (bfcl, gaia, hotpotqa) and on tasks like the training data, there is no gain. Knowing even the real result adds little there.

## Results in brief

| Question | Answer |
|---|---|
| Does a learned selector beat the LLM's own first choice? (ALFWorld) | Yes. The decision model raised useful picks from 28.0% to 41.5% offline, and games won from 63.4% to 72.4% online. |
| Is the JEPA arm better than the decision model? (ALFWorld) | No. It was 3.9 points behind offline and won 6.0 points fewer games online. |
| Does 2-step lookahead through the world model help? (ALFWorld) | No. |
| Does judging the *predicted* result beat judging the call? (AgentProcessBench) | **Yes, on unseen stateful tools (tau2).** CLM heads +0.17 AUC, MLP +0.10. No gain on the lookup-style sources. |
| Does LeJEPA's SIGReg regulariser help? | No. With a frozen encoder it does nothing at small weights and hurts at large ones. |

Practical reading:
- **For stateful tools** (where the response depends on hidden state), put a LeJEPA-style result predictor in front of the judge.
- **For stateless lookup tools,** a decision model that judges the call alone is enough.

Full details, tables and confidence intervals:
- [Implementation Plan.md](Implementation%20Plan.md): ALFWorld, Phases 0–5.
- [Option 3 Plan — LeJEPA predictor + CLM heads.md](Option%203%20Plan%20—%20LeJEPA%20predictor%20+%20CLM%20heads.md): AgentProcessBench, sections 2.3–2.5.
- [JEPA vs Decision Model for Agents — Experiment Plan.md](JEPA%20vs%20Decision%20Model%20for%20Agents%20—%20Experiment%20Plan.md): the original design.

## Setup

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
ALFWORLD_DATA=data/alfworld uv run alfworld-download   # game files go to data/alfworld/json_2.1.1
```

`data/` and `results/` are git-ignored, so every dataset, cache and result is created locally.

**LLM proposer (ALFWorld only).** The proposer can be either:
- Claude, through the Anthropic SDK (the default), using `ANTHROPIC_API_KEY`;
- any OpenAI-compatible endpoint, set with `--provider openai --model M --base-url URL --api-key-env VAR`.

The reported runs used Qwen3.8-27B with reasoning off.

**AgentProcessBench (Option 3 only).**
- **Data:** download `LulaCola/AgentProcessBench` from Hugging Face into `data/agentprocessbench/`, as `bfcl.jsonl`, `gaia_dev.jsonl`, `hotpotqa.jsonl` and `tau2.jsonl`.
- **Encoder:** CLM-v0.1-8B's Qwen3-8B encoder, as a GGUF from `czl/CLM-v0.1-8B-GGUF`. Use the Q8_0 file, not Q4_K_M, which the model card rates poorly for ranking.
- **Heads:** `CLM_v0.1-8B.pt` from `Contrastive-LM/CLM-v0.1-8B`, saved to `data/clm/`.
- **Serving:** run the encoder with llama.cpp, since Ollama does not serve these embeddings:

  ```bash
  llama-server -m CLM-Q8_0.gguf --embedding --pooling last -c 8192 -np 2 -ngl 99 -b 4096 -ub 4096 --port 8090
  ```

## Running

### ALFWorld

```bash
uv run python -m src.phase0_check 10 6                  # Phase 0: environment and label checks
uv run python -m src.collect --run NAME --games-per-type 45 [proposer flags]   # Phase 1: labelled candidates (resumable)
uv run python -m src.collect --run NAME --summary       #   stats, including the headroom check
uv run python -m src.embed --run NAME                   # cache sentence embeddings
uv run python -m src.eval_offline --run NAME --seeds 5  # Phases 2-3: train selectors, offline comparison
uv run python -m src.eval_online --train-run NAME --run ONLINE [proposer flags]  # Phase 4: whole games
uv run python -m src.eval_online --train-run NAME --run ONLINE --summary
uv run python -m src.lookahead --run NAME               # Phase 5: 2-step lookahead
```

Add `--stub` to `collect` or `eval_online` to test the pipeline without an LLM.

### AgentProcessBench (Option 3)

```bash
uv run python -m src.apb                       # build one row per tool-call step
uv run python -m src.option3 --encode-only     # embed with the CLM encoder (slow, cached, resumable)
uv run python -m src.option3 --seeds 3         # train and compare; writes results/apb/option3.json
uv run python -m src.option3 --sigreg 0.001,0.005 --out option3_followup.json
```

On a 16 GB Apple-silicon Mac:
- embedding the 5,064 steps takes about 6 hours;
- once the embeddings are cached, training and evaluation take about 2 hours on the CPU.

## Code layout

```
src/
  env.py, progress.py, tw_speedups.py   ALFWorld wrapper, exact steps-to-go labels, replay speed-ups
  agent.py                              LLM proposers (Anthropic, OpenAI-compatible, stub)
  collect.py                            Phase 1: candidates + counterfactual labels by replay
  embed.py                              frozen all-mpnet-base-v2 embeddings, cached
  models.py                             predictor, value head, decision model, hybrid
  eval_offline.py                       Phases 2-3
  eval_online.py                        Phase 4
  lookahead.py                          Phase 5
  clm_eval.py                           CLM-v0.1-8B encoder client and heads, zero-shot on ALFWorld
  apb.py                                AgentProcessBench to per-step rows
  option3.py                            LeJEPA predictor + CLM heads / MLP judges, AUC comparison
```

## Method notes

**ALFWorld labels.** Every candidate at every step is executed by replay, then labelled from exact steps-to-go:
- *useful*: it gets closer to the goal;
- *error*: the game rejects it;
- *dead end*: otherwise.

**Models.** All models train on frozen embeddings:
- ALFWorld uses all-mpnet-base-v2 (768-d);
- AgentProcessBench uses Qwen3-8B (4096-d).

The JEPA predictor maps (state, action) to the embedding of the outcome, and a judge scores that prediction.

**Splits.**
- ALFWorld is split by game, with held-out task types and held-out verbs.
- AgentProcessBench is split by task, with each of its four sources held out in turn.

**Statistics.** Several seeds; 95% bootstrap intervals over games or tasks; paired differences between selectors.
