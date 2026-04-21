"""Parser registry for LAI tool outputs."""

from popout.benchmark.parsers.flare import parse_flare
from popout.benchmark.parsers.popout import parse_popout
from popout.benchmark.parsers.truth import parse_truth

PARSERS = {
    "flare": parse_flare,
    "popout": parse_popout,
    "truth": parse_truth,
    # "rfmix": parse_rfmix,         # TODO: RFMix .msp.tsv + .Q files
    # "gnomix": parse_gnomix,       # TODO: Gnomix per-window predictions
    # "sparsepainter": ...,         # TODO: SparsePainter chunk output
    # "orchestra": ...,             # TODO: Orchestra ensemble output
    # "recombmix": ...,             # TODO: Recomb-Mix tract output
}


def get_parser(name: str):
    """Look up a parser by tool name."""
    if name not in PARSERS:
        raise ValueError(
            f"No parser registered for {name!r}. "
            f"Available: {sorted(PARSERS.keys())}"
        )
    return PARSERS[name]
