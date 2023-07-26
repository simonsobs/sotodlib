# %%
from sotodlib.io import imprinter

# %%
im = imprinter.Imprinter("/home/yguan/data_pkg/imprinter.yaml")

# %%
books = im.get_books()
# %%
hkbooks = [book for book in books if book.type == "hk"]
#%%
im.bind_book(hkbooks[-1], test_mode=True)
