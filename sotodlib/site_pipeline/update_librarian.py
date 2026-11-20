"""
Upload bound Books to the Librarian, the service that holds the offsite
copies. Every Book with status BOUND is handed to the Librarian and, on
success, moved to status UPLOADED. Only UPLOADED Books are eligible for
deletion by ``cleanup_level2``.

The Librarian connection is named by the ``librarian_conn`` key in the
imprinter configuration file and resolved against the ``hera_librarian``
client settings.

The loop gives up after 5 failed uploads, so a Librarian outage produces a
handful of log lines rather than one per Book, and then re-raises the first
failure so the scheduler sees a non-zero exit status.

See the Data Packaging page of the sotodlib documentation for the full
pipeline description.
"""

import argparse

from sotodlib.io.imprinter import Imprinter
from sotodlib.site_pipeline.utils.logging import init_logger
from sotodlib.site_pipeline.utils.profiler import profile, add_profile_args

logger = init_logger(__name__, "update_librarian: ")


@profile("update_librarian")
def main(config: str):
    """
    Upload all BOUND Books to the Librarian and mark them UPLOADED.

    Stops after 5 failures and re-raises the first one.

    Parameters
    ----------
    config : str
        Path to config file for imprinter. Must set ``librarian_conn``.
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
        parser = argparse.ArgumentParser(
            description="Upload every BOUND Book in the imprinter database to "
            "the Librarian and mark it UPLOADED."
        )
    parser.add_argument(
        '--config', type=str,
        help="imprinter configuration file. Must set 'librarian_conn'."
    )

    add_profile_args(parser)

    return parser


if __name__ == "__main__":
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))
