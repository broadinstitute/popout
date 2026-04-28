"""Regenerate ``tests/data/baseline_no_priors.npz``.

This is the frozen baseline for
:func:`tests.test_determinism_10k.test_priors_none_matches_frozen_baseline`.
Run with::

    python -m tests.gen_baseline_no_priors

from the worktree root. Commit the resulting ``.npz`` file. The test
will then compare the priors=None codepath against this frozen output
on every subsequent test run.

The script duplicates the synthetic-data parameters in
``test_determinism_10k._build_chrom`` and ``_run`` deliberately —
keeping the baseline-generator self-contained means a config drift in
either side is loud, not silent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from popout.em import run_em
from popout.simulate import simulate_admixed


_OUT = Path(__file__).parent / "data" / "baseline_no_priors.npz"


def main() -> None:
    chrom_data, _, _ = simulate_admixed(
        n_samples=750,
        n_sites=500,
        n_ancestries=4,
        gen_since_admix=10,
        chrom_length_cm=80.0,
        rng_seed=4242,
    )
    res = run_em(
        chrom_data,
        n_ancestries=4,
        n_em_iter=3,
        gen_since_admix=10.0,
        rng_seed=0,
        priors=None,
    )
    model = res.model
    payload = {
        "mu": np.array(model.mu),
        "allele_freq": np.array(model.allele_freq),
        "gen_since_admix": np.array(
            float(model.gen_since_admix), dtype=np.float64,
        ),
    }

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(_OUT, **payload)
    print(f"Wrote {_OUT}")
    print(f"  mu shape={payload['mu'].shape}")
    print(f"  allele_freq shape={payload['allele_freq'].shape}")
    print(f"  gen_since_admix={payload['gen_since_admix']}")


if __name__ == "__main__":
    main()
