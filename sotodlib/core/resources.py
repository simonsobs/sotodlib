import os
from urllib.request import urlretrieve

from ast import literal_eval


RESOURCE_DEFAULTS = {
    "de421.bsp": "ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/de421.bsp",
}


def get_local_file(filename: str, cache: bool = True) -> str:
    """
    This function is responsible to check if a resource file exists either in a
    predefined cache or in a user specified path. If not, it tries to download
    from an ftp path.

    Args:
      filename: The name of the file to grab
      cache: A path in the user's local folder to check for existing files and
             cache them after a download

    Returns:
      The aboslute path of the file.
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
            f"File {filename} does not exist in default resources neither in "
            + "$SOTODLIB_RESOURCES. Please set $SOTODLIB_RESOURCES with a "
            + "json where the key is the resource name and value either a "
            + "local path or an ftp path."
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

    return target_file
