"""UniverSat PASTIS-R downstream project."""

# Import the base UniverSat package so that backbone / decode head / data
# preprocessor are registered into MMSegmentation registries.
try:
    import projects.universat.universat  # noqa: F401
except ImportError:  # pragma: no cover
    pass

from .datasets import *  # noqa: F401,F403
from .transforms import *  # noqa: F401,F403
from .utils import *  # noqa: F401,F403
