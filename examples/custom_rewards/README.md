# Custom Rewards — 3-step extension guide

**Goal**: add a new reward function to a forge GRPO job without editing any framework
code.

## Step 1: write the function

```python
# my_custom_reward.py  (anywhere on PYTHONPATH)
from forge.reward import register_reward

@register_reward("exact_match")
def exact_match_reward(
    prompt, completions, prompt_ids=None, completion_ids=None,
    answer=None, **kwargs,
) -> float:
    if answer is None:
        return 0.0
    return 1.0 if str(answer).strip().lower() in str(completions).lower() else 0.0
```

The runtime always passes these five arguments: `prompt`, `completions`, `prompt_ids`,
`completion_ids`, `answer` — plus any dataset-specific extras via `**kwargs`. **You only
need to read the ones you actually use.** For example `gsm8k_reward_fn` looks at only
`completions` and `answer`; the other three arguments are bound but never read. The
signature shape (same names, `**kwargs` at the end) is what matters — use what you need,
ignore the rest.

## Step 2: reference it in YAML

```yaml
# your_grpo_config.yaml
reward: exact_match
```

The short name is what you passed to `@register_reward(...)`.

## Step 3: make sure the module gets imported

Two common options:

### (a) Drop the file under `forge.reward.*` or `areal.reward.*`

Auto-discovery imports every submodule of these two packages at lookup time, so the
decorator fires without further action.

```bash
cp my_custom_reward.py /root/AReaL/forge/reward/
```

### (b) Import from your launch entry script

```python
# your_train.py
import my_custom_reward  # noqa: F401  (decorator side effect)
# ... rest of your training code
```

### (c) External pip package with an entry point

Not wired yet; see the follow-up note in `forge/reward/__init__.py` if you want to add
this. For most forge users, (a) is the recommended workflow.

## Verify

```bash
python -m forge list-rewards
```

You should see your function in the output, grouped by the module it was defined in:

```
Found 4 registered reward(s):

  areal.reward.gsm8k
    • gsm8k        → gsm8k_reward_fn

  forge.reward.exact_match
    • exact_match  → exact_match_reward
  ...
```

If it's missing, the module didn't get imported — revisit Step 3.

(You can also call
`from forge.reward import available_rewards; print(available_rewards())` from any Python
REPL inside the conda env, but the CLI is faster.)

## Backward compatibility

The legacy `reward_fn_path: "my.module.my_fn"` style still works. The short-name path
takes precedence when both are set. Migrating an existing YAML to the new style is a
one-line change.

## Signature conventions

Rewards are called with these keyword arguments. Ignore what you don't need via
`**kwargs`:

| arg              | type        | meaning                            |
| ---------------- | ----------- | ---------------------------------- |
| `prompt`         | `str`       | rendered prompt text               |
| `completions`    | `str`       | model response                     |
| `prompt_ids`     | `list[int]` | tokenizer ids for the prompt       |
| `completion_ids` | `list[int]` | tokenizer ids for the completion   |
| `answer`         | `Any`       | dataset ground truth (often a str) |

Return value: `float` — usually in `[0, 1]` but not enforced.
