"""Seeds that have to survive a new interpreter.

`hash()` of a string is salted per process, so a seed built from
`abs(hash(name))` picks a different number on every run. That silently made the
demo captures and the broken copies different each time the pipeline ran, which
moved the headline Spearman figure by 0.14 between two runs of the same
commit. The subprocess tests below are the point: in one process the old code
looked perfectly reproducible.
"""

from __future__ import annotations

import subprocess
import sys

from capturecheck.utils.seed import stable_seed

SNIPPET = (
    "from capturecheck.utils.seed import stable_seed;"
    "print(stable_seed('kitchen_orbit', 'blur', 'heavy'), stable_seed('kitchen', modulo=10_000))"
)


def _in_new_process(hash_seed: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", SNIPPET],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": hash_seed, "PATH": "/usr/bin:/bin"},
    )
    return result.stdout.strip()


def test_stable_seed_is_the_same_in_a_fresh_interpreter():
    parts = stable_seed("kitchen_orbit", "blur", "heavy")
    scene = stable_seed("kitchen", modulo=10_000)
    assert _in_new_process("0") == f"{parts} {scene}"


def test_stable_seed_ignores_pythonhashseed():
    """The salt that broke `hash()` must not reach this."""
    values = {_in_new_process(seed) for seed in ("0", "1", "12345", "random")}
    assert len(values) == 1


def test_stable_seed_respects_the_modulo():
    for name in ("kitchen", "lobby", "garage", "office", "studio"):
        assert 0 <= stable_seed(name, modulo=10_000) < 10_000


def test_different_parts_give_different_seeds():
    seeds = {
        stable_seed("kitchen_orbit", op, strength)
        for op in ("blur", "noise", "subsample")
        for strength in ("light", "medium", "heavy")
    }
    assert len(seeds) == 9


def test_the_parts_are_not_just_concatenated():
    """('ab', 'c') and ('a', 'bc') are different keys, not the same one."""
    assert stable_seed("ab", "c") != stable_seed("a", "bc")


def test_set_seed_does_not_pretend_to_fix_string_hashing():
    """It used to set PYTHONHASHSEED, which does nothing after start-up."""
    import inspect

    from capturecheck.utils import seed as seed_module

    source = inspect.getsource(seed_module.set_seed)
    assert "PYTHONHASHSEED" not in source.split('"""')[2]
