# Priors

`popout` supports anthropologically-motivated priors on per-component
generations since admixture (T_k). Priors are **emergent**: each prior
carries an *identity signature* describing what a matching ancestry
component should look like, plus a *parameter claim* on T. At every EM
iteration the framework soft-assigns priors to components by similarity,
and the parameter claims apply weighted by the assignment.

This replaces the previous (v1) `component_idx`-based binding, which
was brittle: the index of the AFR-bearing ancestry component depends on
the recursive seeding tree's leaf order, which is not stable across
runs at biobank scale.

## Schema (v2)

A priors YAML looks like:

```yaml
schema_version: 2
morgans_per_step: 0.0001

priors:
  - name: AFR
    identity:
      aims:
        panel: bundled:african.tsv     # path to AIM panel TSV
        weight: 1.0                    # optional, default 1.0
      fst_reference:
        superpop: AFR                  # column name in the 1KG superpop-freqs TSV
        # superpop_freqs_path: <path>  # optional override; default = ~/.popout cache
        weight: 1.0
    parameters:
      gen:
        mean: 7
        range: [4, 12]
    source: "Atlantic slave trade primary phase, Bryc 2015"

  - name: EUR
    identity:
      aims:
        panel: bundled:european.tsv
      fst_reference:
        superpop: EUR
    parameters:
      gen:
        mean: 2
        range: [1, 4]

annealing:
  schedule: linear
  tau_start: 1.0
  tau_end: 0.1
  ramp_iters: 10
```

### Top-level keys

| key                | required | meaning |
|--------------------|----------|---------|
| `schema_version`   | yes      | Must be `2`. v1 (`component_idx`) is rejected. |
| `morgans_per_step` | yes      | Per-step genetic distance — sets the time scale of the Beta(α,β) on r. |
| `priors`           | yes      | Non-empty list of priors. |
| `annealing`        | no       | Annealing schedule for the soft assignment. Default: `linear` 1.0→0.1 over 10 iters. |

### Identity signatures

Each prior must have **at least one** identity signature.

* **`aims`** — variance-normalized weighted L2 against an AIM panel.
  The panel is a TSV with header columns `chrom`, `pos_bp`,
  `expected_freq`, `weight`. Bundled panels live at
  `popout/data/aim_panels/` and can be referenced as
  `panel: bundled:<name>.tsv`.

* **`fst_reference`** — negative Hudson F_ST against a 1KG superpop
  frequency vector. Resolves the superpop-freqs TSV from
  `~/.popout/superpop_freqs/{genome}/1kg_superpop_freq.tsv.gz` by
  default (populate via `popout fetch-superpop-freqs`), via the
  `--superpop-freqs PATH` CLI flag, or via an explicit
  `superpop_freqs_path` in the YAML block.

Add new signature types by writing a frozen dataclass that conforms to
`popout.identity.IdentitySignature`; the dispatcher needs no changes.

### Parameter claims

Currently only `gen` (generations since admixture) is supported.
`gen.mean` is the documented central estimate; `gen.range` is the
[5th, 95th] percentile band of demographic uncertainty. The loader
solves for Beta(α, β) on `r = 1 - exp(-T * morgans_per_step)` so that
the Beta percentiles match.

### Annealing

Soft assignment uses softmax with temperature τ. The annealing
schedule cools τ over the first `ramp_iters` iterations, so priors
gently bias many components early and increasingly target their best
match late.

## Audit artifact

Run with `--priors-dump-assignments PATH.tsv` to write the final
`(P, K)` assignment matrix at run end. Rows are prior names; columns
are component indices with their nearest-1KG-superpop annotation.

This is the diagnostic that catches a prior having latched onto the
"wrong" component, which is what motivated the redesign. Read the
dump first; concordance numbers second.

## Migration from v1

If you have a v1 YAML (`morgans_per_step` + `components: [{component_idx: ..., gen_mean: ..., gen_lo: ..., gen_hi: ...}]`):

1. Replace `component_idx` with structural identity. The new YAML keys
   priors by population *name* (e.g. `AFR`) — no index involved.
2. For each prior, write an `identity:` block. Start with `fst_reference: {superpop: <pop>}` for the 1KG-superpop priors. Add an
   `aims:` block referencing the bundled panel for that population.
3. Move `gen_mean`/`gen_lo`/`gen_hi` into `parameters: { gen: { mean, range: [lo, hi] } }`.
4. Add `schema_version: 2` at the top.
5. (Optional) Add an `annealing:` block. The default is fine for most
   experiments.

The v1 loader (`popout/priors.py`'s `ComponentTPrior` /
index-based `load_priors`) has been removed; loading a v1 YAML errors
out with a pointer back here.

## Known limitations

* Identity scoring uses one chromosome's allele frequencies at a time
  (the EM loop is per-chromosome). Aggregating across chromosomes is
  an additive future change, not breaking.
* The MID superpop is not in the 1KG reference TSV — MID priors rely
  on AIM signatures alone.
* Starter AIM panels are intentionally small (5–10 well-documented
  markers per superpop). Richer panels are an experimentation lever;
  see `popout/data/aim_panels/` for the source of each marker.
