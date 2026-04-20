"""popout — GPU-accelerated self-bootstrapping local ancestry inference.

No reference panel required.  Feed it phased WGS from a large cohort
and ancestry structure falls out of the joint distribution.
"""

import os as _os
__version__ = _os.environ.get("POPOUT_VERSION") or "0.0.0+source"
