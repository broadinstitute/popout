"""popout — GPU-accelerated self-bootstrapping local ancestry inference.

No reference panel required.  Feed it phased WGS from a large cohort
and ancestry structure falls out of the joint distribution.
"""

try:
    from ._version import __version__
except ImportError:
    from importlib.metadata import version as _v
    __version__ = _v("popout")
