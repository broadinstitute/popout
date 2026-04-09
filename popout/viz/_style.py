"""Shared visual identity for popout plots.

Provides a colorblind-safe ancestry palette, GRCh38 chromosome lengths,
and a context manager for publication-quality matplotlib rcParams.
"""

from __future__ import annotations

from contextlib import contextmanager

# Paul Tol qualitative palette — colorblind-safe, up to 12 distinct colors.
# https://personal.sron.nl/~pault/data/colourschemes.pdf
ANCESTRY_PALETTE = [
    "#4477AA",  # blue
    "#EE6677",  # red/rose
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey
    "#EE8866",  # orange
    "#44BB99",  # teal
    "#DDCC77",  # sand
    "#882255",  # wine
    "#332288",  # indigo
]


def ancestry_colors(n: int) -> list[str]:
    """Return the first *n* ancestry colors from the palette."""
    if n > len(ANCESTRY_PALETTE):
        import matplotlib.cm as cm
        return [cm.tab20(i / max(n - 1, 1)) for i in range(n)]
    return ANCESTRY_PALETTE[:n]


# GRCh38 autosome lengths (bp). Source: UCSC Genome Browser.
CHROM_LENGTHS_GRCH38: dict[str, int] = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559,
    "chr4": 190214555, "chr5": 181538259, "chr6": 170805979,
    "chr7": 159345973, "chr8": 145138636, "chr9": 138394717,
    "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
    "chr16": 90338345, "chr17": 83257441, "chr18": 80373285,
    "chr19": 58617616, "chr20": 64444167, "chr21": 46709983,
    "chr22": 50818468,
}

# Canonical chromosome order
CHROM_ORDER = [f"chr{i}" for i in range(1, 23)]


def normalize_chrom(c: str) -> str:
    """Normalize chromosome name to 'chrN' form."""
    c = c.strip()
    if c.startswith("chr"):
        return c
    return f"chr{c}"


def chrom_length(chrom: str) -> int:
    """Return GRCh38 length for a chromosome, or 0 if unknown."""
    return CHROM_LENGTHS_GRCH38.get(normalize_chrom(chrom), 0)


def chrom_sort_key(chrom: str) -> int:
    """Sort key for chromosome ordering."""
    c = normalize_chrom(chrom)
    try:
        return CHROM_ORDER.index(c)
    except ValueError:
        return 99


@contextmanager
def popout_style():
    """Context manager setting publication-quality matplotlib rcParams."""
    import matplotlib.pyplot as plt

    params = {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
    with plt.rc_context(params):
        yield
