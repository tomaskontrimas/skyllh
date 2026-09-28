import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skyllh.datasets import create_datasets

__all__ = [
    'create_datasets',
]

# Initialize top-level logger with a do-nothing NullHandler. It is required to
# be able to log messages when user has not set up any handler for the logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())


def __getattr__(name):
    if name == 'create_datasets':
        from skyllh.datasets import create_datasets

        return create_datasets
    raise AttributeError(f"module 'skyllh' has no attribute {name!r}")
