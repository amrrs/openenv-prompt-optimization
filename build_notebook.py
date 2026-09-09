"""Generate prompt_optimization_openenv.ipynb from source cells.

Keeping the notebook as code makes it easy to regenerate. Run:
    uv run python build_notebook.py
then execute it with:
    uv run jupyter nbconvert --to notebook --execute --inplace prompt_optimization_openenv.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

cells = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md(r"""
# Prompt Optimization with OpenEnv and Open Models

*Tune an IT access-request agent against environment rewards on Nebius Token Factory, no fine-tuning required.*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nebius/token-factory-cookbook/blob/main/integrations/openenv/prompt-optimization/prompt_optimization_openenv.ipynb)

This notebook walks through the tutorial in [README.md](README.md):

1. Build an [OpenEnv](https://github.com/huggingface/OpenEnv) environment that simulates an IT service desk handling access requests.
2. Serve it, connect a typed client, and step through an episode by hand.
3. Plug in an open model from [Nebius Token Factory](https://tokenfactory.nebius.com/) as the policy.
4. Optimize the agent's system prompt against the environment's reward with a stronger Token Factory model as the "reflector".
5. Check the optimized prompt on held-out scenarios and on other models.

Only the prompt changes. The model weights never do, which is exactly why this works with hosted models.
""")

md(r"""
## 0. Setup

You need a Nebius Token Factory API key ([get one here](https://tokenfactory.nebius.com/)).

* **Locally**: `cp env.example .env` and put the key in `.env`, then run the notebook with the project's kernel (`uv sync --all-groups` first).
* **Colab**: add `NEBIUS_API_KEY` to Colab *Secrets* (left sidebar) and enable notebook access, then run the install cell.
""")

code(r"""
import os, sys, subprocess, pathlib

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    # Fetch the tutorial folder and install dependencies
    if not pathlib.Path("prompt-optimization").exists():
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        "https://github.com/nebius/token-factory-cookbook.git", "tfc"], check=True)
        subprocess.run(["git", "-C", "tfc", "sparse-checkout", "set", "integrations/openenv/prompt-optimization"], check=True)
        subprocess.run(["cp", "-r", "tfc/integrations/openenv/prompt-optimization", "."], check=True)
    os.chdir("prompt-optimization")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openenv==0.4.1", "fastmcp", "openai", "python-dotenv", "httpx", "uvicorn", "fastapi"], check=True)
    from google.colab import userdata
    os.environ["NEBIUS_API_KEY"] = userdata.get("NEBIUS_API_KEY")
else:
    from dotenv import load_dotenv
    load_dotenv()

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
assert os.getenv("NEBIUS_API_KEY"), "NEBIUS_API_KEY not found"
print("Token Factory key found; working directory:", os.getcwd())
""")

md(r"""
## 1. The environment

`access_request_env/` is a standard OpenEnv environment package (the layout `openenv init` generates):

```
access_request_env/
├── openenv.yaml                 # manifest used by openenv build / push
├── models.py                    # AccessRequestState (+ re-exported MCP action/observation types)
├── client.py                    # AccessRequestEnv(MCPToolClient)
├── pyproject.toml, README.md    # installable package + Hugging Face Space card
└── server/
    ├── scenarios.py             # deterministic ticket generator + policy engine (ground truth)
    ├── access_request_environment.py  # AccessRequestEnvironment(MCPEnvironment): tools + reward
    ├── app.py                   # FastAPI app via openenv.core create_app
    └── Dockerfile
```

The environment is an **`MCPEnvironment`**: its tools are ordinary Python functions registered on a FastMCP server, and agents act through `CallToolAction`. Five tools are read-only (`get_ticket`, `get_employee`, `get_system_policy`, `check_manager_approval`, and, in easy mode, `get_access_policy`). Three tools are terminal decisions (`grant_access`, `deny_request`, `escalate`) and end the episode with a reward computed from the hidden ground truth.

Let's look at the reward logic and the decision rules the environment enforces.
""")

code(r"""
from access_request_env.server.scenarios import ACCESS_POLICY_TEXT, ARCHETYPES, generate_scenario
print(ACCESS_POLICY_TEXT)
""")

code(r"""
import json
# Seeds are deterministic. 1001 % 13 == 0, so seeds 1001..1013 walk through every scenario archetype once.
for seed in (1001, 1003, 1010):
    s = generate_scenario(seed)
    print(f"seed={seed} archetype={s.archetype}")
    print("  ticket :", json.dumps(s.ticket.__dict__)[:160], "...")
    print("  truth  :", s.ground_truth)
""")

md(r"""
### 1a. Reward

| Outcome | Reward |
|---|---:|
| Correct decision with the right reason code / recipient | +1.0 |
| Correct decision, wrong reason code or recipient | +0.7 |
| Unnecessary escalation | +0.2 |
| Deny when the answer was escalate | 0.0 |
| Deny when the answer was grant | -0.2 |
| Grant to the wrong employee / system / level | 0.0 |
| **Grant when the answer was deny or escalate** | **-1.0** |
| No decision within 12 steps | -0.5 |

minus 0.05 per tool call beyond the first six and 0.1 per invalid tool call. Wrong grants are the security failure, so they dominate the reward. The reward lives entirely inside the environment (`_score_decision` in `access_request_environment.py`); the agent code never sees the ground truth until the episode is over.
""")

md(r"""
## 2. Serve the environment and connect a client

OpenEnv environments run as a FastAPI server (locally, in Docker, or as a Hugging Face Space). `openenv serve` is still a placeholder in OpenEnv 0.4.x, so we start the app directly. The server exposes `/ws` (what the client uses), `/reset`, `/step`, `/state`, `/health`, `/docs` and a small web UI at `/web`.
""")

code(r"""
from env_server import start_env_server, stop_env_server

ENV_PORT = 8010
ENV_URL = f"http://127.0.0.1:{ENV_PORT}"
server = start_env_server(port=ENV_PORT)          # hard mode: policy rules are NOT exposed as a tool
print("environment server running at", ENV_URL, "(web UI at", ENV_URL + "/web)")
""")

code(r"""
from access_request_env import AccessRequestEnv, CallToolAction

# The client is async by default; .sync() gives a blocking wrapper that is convenient in notebooks.
with AccessRequestEnv(base_url=ENV_URL).sync() as env:
    result = env.reset(seed=1003)                  # archetype: terminated requester
    ticket = result.observation.metadata["ticket"]
    print("ticket:", json.dumps(ticket, indent=2))
    print("\ntools:", [t.name for t in env.list_tools()])
    print("state:", env.state())
""")

md(r"""
### 2a. A scripted episode

Every `step()` returns a `StepResult` with `observation`, `reward` and `done`. Investigation steps carry reward 0.0; the decision step carries the final reward and reveals the ground truth in `observation.metadata`.
""")

code(r"""
from access_request_env import tool_payload

with AccessRequestEnv(base_url=ENV_URL).sync() as env:
    result = env.reset(seed=1003)
    ticket = result.observation.metadata["ticket"]

    r = env.step(CallToolAction(tool_name="get_employee", arguments={"employee_id": ticket["requester_id"]}))
    emp = tool_payload(r.observation)
    print("employee:", emp["title"], "|", emp["employment_type"], "|", emp["status"], "| reward:", r.reward, "| done:", r.done)

    r = env.step(CallToolAction(tool_name="deny_request",
                                arguments={"reason_code": "employment_status",
                                           "note": "Rule 1: HR record shows the requester is terminated."}))
    print("decision result:", tool_payload(r.observation))
    print("reward:", r.reward, "| done:", r.done, "| outcome:", r.observation.metadata["outcome"])
    print("ground truth:", r.observation.metadata["ground_truth"])
""")

md(r"""
## 3. A Token Factory model as the policy

`agent.py` contains the whole agent loop in one function, `run_episode`:

1. `env.reset(seed=...)` and read the ticket.
2. Convert the environment's MCP tool manifest (`env.list_tools()`) to OpenAI function-calling schemas.
3. Call the Token Factory chat completions API (OpenAI-compatible) with the system prompt, the ticket and the tools.
4. For each tool call the model makes, `env.step(CallToolAction(...))` and feed the result back as a `tool` message.
5. Stop when the environment says `done`.

`run_batch` runs many seeds in parallel, each on its own WebSocket session (the server allows 16 concurrent sessions by default).

We start with a deliberately minimal system prompt, the kind of thing a team writes on day one.
""")

code(r"""
from agent import load_prompt, run_episode, run_batch, train_seeds, holdout_seeds, DEFAULT_POLICY_MODEL

POLICY_MODEL = DEFAULT_POLICY_MODEL   # zai-org/GLM-5.3-Flash by default; override with POLICY_MODEL env var
baseline_prompt = load_prompt("baseline")
print("policy model:", POLICY_MODEL)
print("baseline prompt:", repr(baseline_prompt))
""")

code(r"""
ep = run_episode(baseline_prompt, seed=1010, model=POLICY_MODEL, env_url=ENV_URL)   # archetype: role_exception
print("scenario:", ep.scenario_type, "| reward:", ep.reward, "| outcome:", ep.outcome, "| llm calls:", ep.llm_calls)
for call in ep.tool_calls:
    print("  ->", call["tool"], json.dumps(call["arguments"])[:100])
print("agent decision :", ep.decision)
print("ground truth   :", ep.ground_truth)
""")

code(r"""
# One pass over the 13 archetypes with the baseline prompt
baseline_13 = run_batch(baseline_prompt, train_seeds(13), model=POLICY_MODEL, env_url=ENV_URL, workers=8, prompt_name="baseline")
for e in baseline_13.episodes:
    print(f"  seed={e.seed} {e.scenario_type:<26} reward={e.reward:+.2f} {e.outcome}")
""")

md(r"""
## 4. Optimize the prompt against the reward

`optimize.py` implements a small reflective loop (the idea behind optimizers such as GEPA, written out in about 100 lines so you can see every step):

1. Evaluate the current best prompt on the training seeds.
2. Collect the imperfect episodes: ticket, tool trace, the agent's decision and the ground truth the environment revealed.
3. Ask a stronger Token Factory model (the **reflector**, Kimi K3 by default) to diagnose the failures and write a better system prompt. The reflector must keep the prompt general (no ticket ids or names) and may only reference tools that exist.
4. Evaluate the candidate on the same seeds. Keep it if the mean reward improves.
5. Finally compare baseline and best prompt on **held-out seeds** the optimizer never saw.

Why the environment matters here: seeded `reset()` gives identical scenario batches for every candidate, `StepResult.reward` gives a trajectory-level score with no separate grader to build, and the sandbox means we can burn through hundreds of episodes with zero risk to real systems.

The cell below uses a small configuration (13 seeds, one round, one candidate) so it finishes in about ten minutes with GLM-5.3-Flash. The numbers in the README come from the full CLI run (`python optimize.py --n-train 26 --n-holdout 26 --rounds 3 --candidates 2`).
""")

code(r"""
from optimize import optimize
from pathlib import Path

run = optimize(
    baseline_prompt,
    policy_model=POLICY_MODEL,
    reflector_model="moonshotai/Kimi-K3",
    env_url=ENV_URL,
    n_train=13, n_holdout=13,
    rounds=1, candidates_per_round=1,
    workers=8,
    out_dir=Path("results/notebook_run"),
    save_prompt_to=Path("results/notebook_run/optimized.md"),
)
""")

code(r"""
print("=== held-out seeds (never seen by the optimizer) ===")
for name, s in (("baseline", run.baseline_holdout), ("optimized", run.best_holdout)):
    print(f"{name:<10} reward={s['mean_reward']:.3f} accuracy={s['accuracy']:.0%} "
          f"unauthorized_grants={s['unauthorized_grants']} unnecessary_escalations={s['unnecessary_escalations']} "
          f"tool_calls={s['mean_tool_calls']:.1f}")
""")

md(r"""
A note on sample size: 13 seeds is one ticket per scenario type, so a single episode moves the mean reward by 0.077, and the baseline agent only fails on one or two of them. A one-round, one-candidate run can therefore look flat or even slightly worse on held-out seeds. The full configuration in the README (26 training seeds, 2 candidates per round, up to 3 rounds) is what produced `prompts/optimized.md`, which scores 1.000 on 26 held-out seeds. We compare against that prompt in the next section.
""")

code(r"""
print(run.best_prompt)
""")

md(r"""
## 5. Does the prompt transfer to other models?

An enterprise team rarely stays on one model. Because the prompt is the only artifact, re-checking it on another Token Factory model is one function call. We compare the baseline, the prompt this notebook just produced, and the prompt from the full run shipped in the repo. `evaluate.py` does this for any set of prompts and models and writes a markdown table.
""")

code(r"""
from evaluate import evaluate, to_markdown

prompts = {
    "baseline": baseline_prompt,
    "optimized (this notebook run)": run.best_prompt,
    "optimized (full run, prompts/optimized.md)": load_prompt("optimized"),
}
models = [POLICY_MODEL, "MiniMaxAI/MiniMax-M3", "Qwen/Qwen3-30B-A3B-Instruct-2507"]
summaries = evaluate(prompts, models, seeds=holdout_seeds(13), env_url=ENV_URL, workers=8)
from IPython.display import Markdown, display
display(Markdown(to_markdown(summaries)))
""")

md(r"""
## 6. Optional: a Token Factory model as an LLM judge inside the environment

OpenEnv has a `Rubric` abstraction for reward computation, including `LLMJudge`. `NoteQualityJudge` in `access_request_environment.py` subclasses it to score the agent's decision note (does it cite the rule and the evidence?) and adds up to +0.2 reward. It uses OpenEnv's `OpenAIClient` pointed at Token Factory. Start the server with `ACCESS_ENV_JUDGE_MODEL` set to enable it:
""")

code(r"""
stop_env_server(server)
judge_server = start_env_server(port=ENV_PORT, judge_model="zai-org/GLM-5.3-Flash")

ep = run_episode(load_prompt("optimized"), seed=5007, model=POLICY_MODEL, env_url=ENV_URL)
print("judge score:", ep.judge_score, "| judge bonus:", ep.judge_bonus, "| total reward:", ep.reward)
print("note:", (ep.decision or {}).get("note"))
stop_env_server(judge_server)
""")

md(r"""
## 7. Ship the environment

The environment is a normal OpenEnv package, so the rest of the OpenEnv toolchain applies:

```bash
cd access_request_env
openenv validate                 # checks manifest, entry point and Docker readiness
openenv build                    # builds the Docker image (openenv-access_request_env:latest)
docker run -p 8000:8000 openenv-access_request_env:latest
openenv push --repo-id <user>/access-request-env   # publish as a Hugging Face Space
```

For enterprise data you would run the same image privately (a Nebius VM or Kubernetes) instead of a public Space.

## 8. Where to go next

* **Change the policy, re-optimize, redeploy.** Edit `ACCESS_POLICY_TEXT` and `decide()` in `scenarios.py`, run `optimize.py` again. No training job.
* **Reinforcement learning.** The same environment plugs into TRL's GRPO trainer through OpenEnv's integration, with a Token Factory model as the judge. See the [OpenEnv docs](https://huggingface.co/docs/openenv).
* **Distillation.** Use the optimized prompt with a strong model to sample high-reward trajectories, then fine-tune a smaller model on Token Factory ([fine-tuning example](../../../post-training/fine-tuning-1/README.md)).
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
    c["source"] = c["source"].splitlines(keepends=True)
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None
Path("prompt_optimization_openenv.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote prompt_optimization_openenv.ipynb with", len(cells), "cells")
