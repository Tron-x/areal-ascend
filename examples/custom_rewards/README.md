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

Signature convention matches existing rewards (`areal.reward.gsm8k.gsm8k_reward_fn` et
al): positional `prompt, completions, prompt_ids, completion_ids, answer`, plus
`**kwargs` to be forward-compatible with new dataset fields.

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

```python
from forge.reward import available_rewards
print(available_rewards())
# ['clevr_count_70k', 'exact_match', 'geometry3k', 'gsm8k']
```

If `exact_match` shows up, your function is reachable. Your next `forge launch` with
`reward: exact_match` in YAML will use it.

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
