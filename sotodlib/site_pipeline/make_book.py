from typing import Optional
import typer
import traceback

from ..io.imprinter import Imprinter


def main(
    im_config: str,
    output_root: str,
    source: Optional[str],
    ):
    """Make books based on imprinter db
    
    Parameters
    ----------
    im_config : str
        path to imprinter configuration file
    output_root : str
        root path of the books 
    source: str, optional
        data source to use, e.g., sat1, latrt, tsat. If None, use all sources
    """
    imprinter = Imprinter(im_config, db_args={'connect_args': {'check_same_thread': False}})
    # get unbound books
    unbound_books = imprinter.get_unbound_books()
    failed_books = imprinter.get_failed_books()
    if source is not None:
        unbound_books = [book for book in unbound_books if book.tel_tube == source]
        failed_books = [book for book in failed_books if book.tel_tube == source]
    print(f"Found {len(unbound_books)} unbound books and {len(failed_books)} failed books")
    for book in unbound_books:
        print(f"Binding book {book.bid}")
        try:
            imprinter.bind_book(book, output_root=output_root)
        except Exception as e:
            print(f"Error binding book {book.bid}: {e}")
            print(traceback.format_exc())

    print("Retrying previously failed books") 
    for book in failed_books:
        print(f"Binding book {book.bid}")
        try:
            imprinter.bind_book(book, output_root=output_root)
        except Exception as e:
            print(f"Error binding book {book.bid}: {e}")
            print(traceback.format_exc())
            # it has failed twice, ideally we want people to look at it now
            # do something here

if __name__ == '__main__':
    typer.run(main)
