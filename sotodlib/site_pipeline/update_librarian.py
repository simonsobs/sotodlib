import argparse
import datetime as dt
from pathlib import Path
from typing import Optional

from sotodlib.io.imprinter import Imprinter, BOUND, UPLOADED
from sotodlib.site_pipeline.utils.logging import init_logger

logger = init_logger(__name__, "update_librarian: ")

def main(config: Optional[str] = None, profile: bool=False,
         profile_output: Optional[Path]=None):
    """
    Arguments
    ---------
    config: string
        configuration file for imprinter
    profile: bool
        if True, will run the script with pyinstrument and output to profile_output
    profile_output: str
        if profile is True, the file name of the directory
        to output the pyinstrument profiling results to
    """

    if profile:
        import pyinstrument
        filename = f"update_librarian_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        output_filename = profile_output / filename if profile_output is not None else filename
        profiler = pyinstrument.Profiler()
        profiler.start()
    
    try:
        core(config=config)
    finally:
        if profile:
            profiler.stop()
            if profile_output is not None:
                with open(output_filename, "w") as f:
                    f.write(profiler.output_html())

def core(config: str):
    """
    Update the book plan database with new data from the g3tsmurf database.

    Parameters
    ----------
    config : str
        Path to config file for imprinter
    """

    imprinter = Imprinter(
        config, 
        db_args={'connect_args': {'check_same_thread': False}},
    )

    session = imprinter.get_session()
    to_upload = imprinter.get_bound_books(session=session)

    failed_list = []
    for book in to_upload:
        success, err = imprinter.upload_book_to_librarian(
            book, session=session, raise_on_error=False
        )
        if not success:
            failed_list.append( (book.bid, err) )
        ## don't just continually fail
        if len(failed_list) > 5:
            break
    
    if len(failed_list) != 0:
        # raise the first error so we know something is wrong
        logger.error(f"Failed to upload books {[f[0] for f in failed_list]}")
        raise failed_list[0][1]


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, help="imprinter configuration file"
    )
    parser.add_argument("--profile", help="Run with pyinstrument profiling", action="store_true")
    parser.add_argument("--profile-output", help="Directory to output pyinstrument profiling results to, if --profile is set", 
                        type=Path)
    return parser


if __name__ == "__main__":
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))
