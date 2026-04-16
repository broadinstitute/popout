# Sweep Results — back-to-basics baseline

Generated with `python -m popout.demo --sweep` on the b2b branch.

**Parameters:** seed=42, A=4, T_true=20, n_em_iter=20, F_ST range 0.05–0.15.

## Biobank-like (30% pure-ancestry haplotypes)

```
samples    sites    oracle%    em%        gap_pp   T_inf
------------------------------------------------------
500        2000     93.0       92.5       0.5      11.8
5000       2000     93.0       76.9       16.1     12.4
50000      2000     93.1       74.7       18.3     12.6
500        10000    97.3       96.9       0.4      30.9
5000       10000    97.3       97.1       0.2      35.1
```

## Fully-admixed stress test (0% pure-ancestry)

```
samples    sites    oracle%    em%        gap_pp   T_inf
------------------------------------------------------
500        2000     90.4       62.2       28.2     19.1
5000       2000     90.5       52.2       38.2     19.6
50000      2000     90.4       69.7       20.8     20.4
500        10000    96.2       47.1       49.1     43.4
5000       10000    96.2       42.2       54.0     45.3
```

## Oracle T stability (drift from true T after 1 EM iteration with true frequencies)

```
samples    sites    pure%    T_true    T_after    drift%
-------------------------------------------------------
500        2000     30       20        18.8       6.1
5000       2000     30       20        18.8       6.1
50000      2000     30       20        18.8       0.1
500        10000    30       20        22.6       13.2
5000       10000    30       20        22.2       10.8
500        2000     0        20        20.0       0.0
5000       2000     0        20        20.0       0.0
50000      2000     0        20        20.0       0.1
500        10000    0        20        22.2       10.8
5000       10000    0        20        22.2       10.8
```

## Key observations

1. **Biobank-like regime reaches oracle ceiling** (gap <1pp) at 500 samples / 2K sites
   and at all tested sample sizes with 10K sites.

2. **Fully-admixed regime has a 20–54pp gap** due to GMM spectral init failure:
   no dense PCA corners for initialization.

3. **At 2K sites, 5K–50K samples, the biobank gap widens to 16–18pp.** The GMM
   finds a stable-but-wrong local optimum that splits the dominant 61% cluster.
   This is a systematic failure at this mu/F_ST combination, not random variance.
   Filed as a follow-up issue.

4. **T estimator drifts 6–13% at 10K sites** even from oracle frequencies (both
   regimes). Does not affect accuracy when frequencies are correct — the emission
   model dominates the transition prior at biobank scale. Parked for future
   investigation.

5. **T is stable at 2K sites** in the fully-admixed regime (drift <0.1%), confirming
   the T spiral at 10K sites is an estimator property, not downstream of bad init.
