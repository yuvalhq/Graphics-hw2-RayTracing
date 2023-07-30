import sys
from typing import IO, Any, Iterator, Optional, Sized, cast


def progressbar(
    it: Iterator[Any],
    *,
    count: Optional[int] = None,
    prefix: Optional[str] = "",
    size: int = 60,
    out: IO = sys.stdout,
):
    if count is None:
        if isinstance(it, Sized):
            count = len(it)
        else:
            raise ValueError(
                "The iterator should have the len function implemented if count=0"
            )

    def show_progress(j: int):
        x = int(size * j / cast(int, count))
        print(
            f"{prefix}[{'#' * x}{'.' * (size - x)}] {j}/{count}",
            end="\r",
            file=out,
            flush=True,
        )

    show_progress(0)
    for i, item in enumerate(it):
        yield item
        show_progress(i + 1)
    print("\n", flush=True, file=out)
