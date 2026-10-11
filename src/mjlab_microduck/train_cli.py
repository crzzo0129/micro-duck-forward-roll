"""`train` console script with GPU argument normalization for mjlab 1.3.

`--hf-jobs` is intercepted from the `mjlab.tasks` plugin entry point
(train_hook.py). This wrapper additionally converts numeric GPU arguments to
the Python collection syntax required by mjlab's tyro configuration.

Why keep it anyway: same-name console scripts are last-writer-wins, and after
an install BOTH dist-info RECORDs claim `.venv/bin/train`. Removing this
declaration therefore does not hand the name back to mjlab — it makes the next
`uv sync` UNINSTALL the file (ours, per our RECORD) while nothing reinstalls
mjlab's. `train` then vanishes from the venv entirely and `uv run train` falls
through to whatever `train` sits on PATH: on one machine liblinear's, which
answered `can't open input file Mjlab-Velocity-Flat-MicroDuck` (2026-08-31).

Keep the declaration. Use this module explicitly for cloud commands so its
normalization is applied regardless of which package owns bin/train.
"""

from __future__ import annotations

import ast
import sys


def normalize_gpu_argv(argv: list[str]) -> list[str]:
    """Accept space-separated IDs; mjlab's tyro config requires one list literal."""
    result = []
    seen = False
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg != "--gpu-ids" and not arg.startswith("--gpu-ids="):
            result.append(arg)
            index += 1
            continue
        if seen:
            raise ValueError("Specify --gpu-ids only once")
        seen = True
        if arg.startswith("--gpu-ids="):
            values = [arg.split("=", 1)[1]]
            index += 1
        else:
            index += 1
            values = []
            while index < len(argv) and not argv[index].startswith("--"):
                values.append(argv[index])
                index += 1
        if values in (["all"], ["None"]):
            value = values[0]
        else:
            if len(values) == 1 and values[0].startswith("["):
                try:
                    ids = ast.literal_eval(values[0])
                except (ValueError, SyntaxError) as exc:
                    raise ValueError("Invalid GPU list literal") from exc
            elif values and all(v.isascii() and v.isdigit() for v in values):
                ids = [int(v) for v in values]
            else:
                raise ValueError("--gpu-ids expects all, None, or GPU indices such as 0 1 2 3")
            if (not isinstance(ids, list) or not ids or
                    any(type(v) is not int or v < 0 for v in ids) or len(set(ids)) != len(ids)):
                raise ValueError("GPU indices must be a non-empty list of distinct non-negative integers")
            value = repr(ids)
        result.extend(["--gpu-ids", value])
    return result


def main() -> int | None:
    # This import runs mjlab's plugin loader, which imports
    # mjlab_microduck.tasks -> train_hook.maybe_submit_to_hf_jobs(). A
    # `--hf-jobs` invocation submits and exits inside the import below; it
    # never comes back here.
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *normalize_gpu_argv(original_argv[1:])]
        from mjlab.scripts.train import main as mjlab_train_main
        return mjlab_train_main()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    sys.exit(main())
