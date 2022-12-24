import typer
from ..io.imprinter import Imprinter


def main(
    im_config: str,
    g3tsmurf_config: str,
    output_root: str,
    ):
    """Make books based on imprinter db
    
    Parameters
    ----------
    im_config : str
        path to imprinter configuration file
    g3tsmurf_config : str
        path to the g3tsmurf db configuration file
    output_root : str
        root path of the books 
    """
    imprinter = Imprinter(im_config, g3tsmurf_config)
    # get unbound books
    unbound_books = imprinter.get_unbound_books()
    failed_books = imprinter.get_failed_books()
    print(f"Found {len(unbound_books)} unbound books and {len(failed_books)} failed books")
    for book in unbound_books:
        print(f"Binding book {book.bid}")
        try:
            imprinter.bind_book(book, output_root=output_root)
        except Exception as e:
            print(f"Error binding book {book.bid}: {e}")

    print("Retrying previously failed books") 
    for book in failed_books:
        print(f"Binding book {book.bid}")
        try:
            imprinter.bind_book(book, output_root=output_root)
        except Exception as e:
            print(f"Error binding book {book.bid}: {e}")
            # it has failed twice, ideally we want people to look at it now
            # do something here

if __name__ == '__main__':
    typer.run(main)
