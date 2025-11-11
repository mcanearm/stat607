import logging
import subprocess
import timeit
from functools import wraps


logger = logging.getLogger(__name__)


def time_fn(fn):
    """Time a function call."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = fn(*args, **kwargs)
        end = timeit.default_timer()
        return result, end - start

    return wrapper


def get_git_hash(nchars=8) -> str:
    """
    Return the current git commit hash as a string.
    GPT Generated
    """

    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")[:nchars]
        )
        return git_hash
    except Exception as e:
        logger.warning(f"Could not get git hash: {e}")
        return "unknown"
