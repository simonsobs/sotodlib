"""
A decorator for profiling 'main' functions. It adds three arguments: profile
(bool), profile_type (str), and profile_output (Path), the directory to write
profiles to.  The decorator itself requries one main argument, the base filename
to write out the profiles as.
"""

import functools
import datetime as dt
import glob
from pathlib import Path
from typing import Optional, Callable
from argparse import ArgumentParser
import sotodlib


def profile(base_filename: str) -> Callable:
    """
    Decorator for profiling 'main' functions in the data-package and
    site-pipeline stack.

    Adds three arguments to the decorated function:
    - profile (bool): Enable profiling
    - profile_type (str): Type of profile output ('html', 'txt', etc.)
    - profile_output (Path): Directory to write profiles to

    These can be added to the parser using the complimentary `add_profile_args`
    function also defined in this file.

    Args:
        base_filename: Base filename to write profiles as (without extension)

    Returns:
        Decorated function that handles profiling
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(
            *args,
            profile: bool = False,
            profile_type: str = "html",
            profile_output: Optional[Path] = None,
            **kwargs,
        ):
            if not profile:
                return func(*args, **kwargs)

            timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H%M%S")
            filename = f"{base_filename}_{timestamp}.{profile_type}"
            output_filename = (
                profile_output / filename
                if profile_output is not None
                else filename
            )

            if profile_type in ["lprof"]:
                import line_profiler as lprof

                filename = f"{base_filename}_{timestamp}"

                profiler = lprof.LineProfiler()
                profiler.add_module(sotodlib, scoping_policy="descendants")

                profiler.add_callable(func)
                wrapped = profiler.wrap_callable(func)

                try:
                    return wrapped(*args, **kwargs)
                finally:
                    with open(output_filename, "w") as f:
                        profiler.print_stats(f, stripzeros=True, rich=True)

            elif profile_type in ["html", "txt"]:
                import pyinstrument


                profiler = pyinstrument.Profiler()
                profiler.start()

                try:
                    return func(*args, **kwargs)
                finally:
                    if profile:
                        profiler.stop()
                        if profile_output is not None:
                            profile_output.mkdir(parents=True, exist_ok=True)

                        if profile_type.lower() == "html":
                            with open(output_filename, "w") as f:
                                f.write(profiler.output_html())
                        elif profile_type.lower() == "txt":
                            with open(output_filename, "w") as f:
                                f.write(str(profiler.output_text()))
                        else:
                            # Default to HTML for unknown types
                            with open(output_filename, "w") as f:
                                f.write(profiler.output_html())
            else:
                raise ValueError(f"Invalid profile type {profile_type}")

        return wrapper

    return decorator


def add_profile_args(parser: ArgumentParser) -> ArgumentParser:
    """
    Adds the required arguments for profiling functions to the parser.
    """
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument(
        "--profile-type",
        choices=["html", "txt", "lprof"],
        default="html",
        help="Type of profile output",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        help="Directory to write profiles to",
        default=Path("/data/data-package/profiles"),
    )

    return parser
