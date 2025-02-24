import pytest
from unittest.mock import create_autospec
import tempfile
import yaml
import os
import datetime

from sotodlib.io.imprinter import *
from sotodlib.io.g3tsmurf_db import Files

from ._helpers import mpi_multi


# These tests are serial-only.
if mpi_multi():
    pytest.skip(allow_module_level=True)


@pytest.fixture(scope="session", autouse=True)
def imprinter():
    with tempfile.NamedTemporaryFile(mode='w') as f:
        g3tsmurf_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        g3tsmurf_config = {
            "data_prefix": "/data",
            "db_path": "test_g3tsmurf.db"
        }
        yaml.dump(g3tsmurf_config, g3tsmurf_file)
        g3tsmurf_file.close()
        
        im_config = {
            'db_path': '_pytest_imprinter.db',
            'g3tsmurf': g3tsmurf_file.name,
            'output_root': 'test_output',
            'tel_tubes': {
                'lat': {
                    'slots': ['slot1', 'slot2']
                },
            }
        }
        yaml.dump(im_config, f)
        f.flush()
        im = Imprinter(f.name, make_db=True)
    return im

# use this to clean up db file afterwards
@pytest.fixture(scope="session", autouse=True)
def delete_imprinter_db(imprinter):
    yield
    if os.path.exists(imprinter.db_path):
        os.remove(imprinter.db_path)
    if os.path.exists('test_g3tsmurf.db'):
        os.remove('test_g3tsmurf.db')

@pytest.fixture
def obsset():
    t0 = 1674090159
    t1 = 1674090259
    dt_t0 = datetime.datetime.fromtimestamp(t0, tz=datetime.timezone.utc)
    dt_t1 = datetime.datetime.fromtimestamp(t1, tz=datetime.timezone.utc)

    file = create_autospec(Files)
    file.start = dt_t0
    file.stop = dt_t1
    file.n_channels = 10
    
    obs1 = create_autospec(spec=G3tObservations, instance=True)
    obs1.obs_id = f"oper_slot1_{t0}" 
    obs1.files = [file] * 2
    obs1.timing = True
    obs2 = create_autospec(spec=G3tObservations, instance=True)
    obs2.obs_id = f"oper_slot2_{t0}"
    obs2.files = [file] * 2
    obs2.timing = True
    obsset = ObsSet([obs1, obs2], mode="oper", slots=["slot1", "slot2", "slot3"], tel_tube="lat")
    return obsset

def test_ObsSet(obsset):
    assert obsset.mode == "oper"
    assert obsset.slots == ["slot1", "slot2", "slot3"]
    assert obsset.tel_tube == "lat"
    assert obsset.obs_ids == ["oper_slot1_1674090159", "oper_slot2_1674090159"]
    assert obsset.get_id() == "oper_1674090159_lat_110"
    assert obsset.contains_stream("slot1") == True
    assert obsset.contains_stream("slot2") == True
    assert obsset.contains_stream("slot3") == False

def test_register_book(imprinter, obsset):
    book = imprinter.register_book(obsset)
    assert book.bid == "oper_1674090159_lat_110"
    assert book.status == UNBOUND
    assert book.obs[0].obs_id == "oper_slot1_1674090159"
    assert book.obs[1].obs_id == "oper_slot2_1674090159"
    assert book.tel_tube == "lat"
    assert book.type == "oper"
    expected_start = (
        datetime.datetime.fromtimestamp(1674090159, tz=datetime.timezone.utc)
        .replace(tzinfo=None)
    )
    expected_stop = (
        datetime.datetime.fromtimestamp(1674090259, tz=datetime.timezone.utc)
        .replace(tzinfo=None)
    )
    assert book.start == expected_start
    assert book.stop == expected_stop
    assert book.max_channels == 10
    assert book.message == ""
    assert book.slots == "slot1,slot2"
    assert imprinter.book_bound("oper_1674090159_lat_110") == False
    with pytest.raises(BookExistsError):
        imprinter.register_book(obsset)

def test_book_exists(imprinter):
    assert imprinter.book_exists("oper_1674090159_lat_110") == True
    assert imprinter.book_exists("oper_1674090159_lat_111") == False

def test_register_book_raises_BookExistsError(obsset, imprinter):
    # try to register the same ObsSet again, should raise BookExistsError
    with pytest.raises(BookExistsError):
        assert imprinter.register_book(obsset)

def test_get_session(imprinter):
    session = imprinter.get_session()
    assert session is not None

def test_get_book(imprinter):
    book = imprinter.get_book('oper_1674090159_lat_110')
    assert book is not None
    assert book.bid == 'oper_1674090159_lat_110'
    assert book.status == UNBOUND

def test_get_unbound_books(imprinter):
    # Create a new unbound book and add it to the database
    unbound_books = imprinter.get_unbound_books()
    # Assert that the book we just created is in the list of unbound books
    assert any([book.bid == 'oper_1674090159_lat_110' and book.status == UNBOUND for book in unbound_books])
    unbound_books[0].status = BOUND
    imprinter.session.commit()
    unbound_books = imprinter.get_unbound_books()
    # Assert that the book we just created is NOT in the list of unbound books
    assert all([book.bid != 'oper_1674090159_lat_110' or book.status != UNBOUND for book in unbound_books])

def test_book_bound(imprinter):
    books = imprinter.get_books()
    assert imprinter.book_bound(books[0].bid) == True
    books[0].status = UNBOUND
    imprinter.session.commit()
    assert imprinter.book_bound(books[0].bid) == False

def test_stream_timestamp():
    obs_id = 'oper_stream1_1674090159'
    stream, timestamp = stream_timestamp(obs_id)
    assert stream == 'stream1'
    assert timestamp == '1674090159'
