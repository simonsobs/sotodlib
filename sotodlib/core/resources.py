import os
from urllib.request import urlretrieve

from ast import literal_eval


RESOURCE_DEFAULTS = {
    "de421.bsp": "ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/de421.bsp",
}


def get_local_file(filename: str, cache: bool = True) -> str:
    """
    This function utilizes RESOURCE_DEFAULTS or SOTODLIB_RESOURCES environment
    variable to manage resource files such as a planet catalog from NASA.

    RESOURCE_DEFAULTS and SOTODLIB_RESOURCES are dictionaries with key a filename
    and value either an FTP or absolute path to the file. The FTP path starts with
    "ftp://" and an absolute path with "file://". For example, a valid value of
    RESOURCE_DEFAULTS or SOTODLIB_RESOURCES is:

    {
    "de421.bsp": "ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/de421.bsp",
    "de422.bsp": "file:///scratch/gpfs/SIMONSOBS/de422.bsp"
    }

    **Note**: When setting SOTODLIB_RESOURCES use a serialized version of the
              dictionary.

    Upon call, this function checks for the filename first in SOTODLIB_RESOURCES,
    and then at the RESOURCE_DEFAULTS. When the key value is an FTP URL, the
    function checks for the file in a predefined path under the user's home
    folder, and returns its path when it exists, otherwise it downloads it.
    If the key value is a path it just returns it, and it assumes that the user
    ahs verified that the file exists.

    The parameter cache sets the folder of the where a file will be downloaded.
    When True, files are stored under the user's home folder. When cache is set
    to False, the files are downloaded under the tmp filesystem and their long
    term storage is not guaranteed.

    Args:
      filename: The name of the file to grab
      cache: A boolean indicating that downloaded files should be cached in user
              home folder (~/.sotodlib/filecache/).

    Returns:
      The absolute path of the file.

    Exceptions:
        RuntimeError: when the requested resource file does not exist as a key
                      in SOTODLIB_RESOURCES env variable or RESOURCE_DEFAULTS.
        RuntimeErorr: When the value of a key is not an ftp or file path.
    """

    # Local cache. This is per user, however we may want to try and right in
    # the path this file is first and then the user's home directory.
    local_cache = os.path.join(os.path.expanduser("~"), ".sotodlib/filecache/")

    env_resource_paths = literal_eval(os.environ.get("SOTODLIB_RESOURCES", "{}"))

    de_url = env_resource_paths.get(filename, None)

    if de_url is None:
        de_url = RESOURCE_DEFAULTS.get(filename, None)

    if de_url is None:
        raise RuntimeError(
            f"File {filename} not registered in RESOURCE_DEFAULTS nor in "
            + "$SOTODLIB_RESOURCES. Please ensure the URL to the file is "
            + "in RESOURCE_DEFAULTS, and ask your system maintainer "
            + "to expose a cached copy through $SOTODLIB_RESOURCES."
        )

    if de_url.startswith("ftp://"):
        # Check if file is already there.
        file_cached = os.path.exists(os.path.join(local_cache, filename))

        if file_cached:
            return os.path.join(local_cache, filename)

        # Set the target path for the file and create it if it does not exist
        target_path = local_cache if cache else "/tmp/"
        os.makedirs(name=target_path, exist_ok=True)
        target_file = os.path.join(target_path, filename)
        _, headers = urlretrieve(de_url, target_file)
    elif de_url.startswith("file://"):
        target_file = de_url[7:]
    else:
        raise RuntimeError(
            f"Malformed resource file URL {de_url}. Resource file URL needs to "
            + 'start with "ftp://" or "file://"'
        )

    return target_file
