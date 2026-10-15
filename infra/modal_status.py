"""Status of spawned training calls listed in outputs/modal_calls.json (run name -> call id)."""

import json
from pathlib import Path

import modal

for run, cid in json.loads(Path("outputs/modal_calls.json").read_text()).items():
    try:
        modal.FunctionCall.from_id(cid).get(timeout=0)
        print(run, "done")
    except TimeoutError:
        print(run, "running")
    except Exception as e:  # noqa: BLE001
        print(run, "failed", type(e).__name__, str(e)[-300:].replace("\n", " "))
