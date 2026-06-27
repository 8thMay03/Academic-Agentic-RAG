from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


@contextmanager
def elapsed_timer() -> Iterator[callable]:
    start = perf_counter()
    yield lambda: perf_counter() - start

