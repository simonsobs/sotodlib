from collections import OrderedDict as odict
import yaml
import os

import sotodlib

class Context(odict):
    def __init__(self, filename=None, site_file=None, user_file=None):
        super().__init__()
        # Start with site and user config.
        site_ok, site_file, site_cfg = _read_cfg(
            site_file, 'SOTODLIB_SITECONFIG',
            os.path.join(os.getcwd(), 'site.yaml'))
        user_ok, user_file, user_cfg = _read_cfg(
            user_file, 'SOTODLIB_USERCONFIG',
            os.path.expanduser('~/.sotodlib.yaml'))

        self.update(site_cfg)
        self.update_context(user_cfg)

        ok, filename, context_cfg = _read_cfg(filename)

        self.update_context(context_cfg)

        self.site_file = site_file
        self.user_file = user_file
        self.filename = filename

    def update_context(self, new_stuff):
        appendable = ['metadata']
        mergeable = ['tags']
        
        for k, v in new_stuff.items():
            if k in appendable and k in self:
                self[k].extend(v)
            elif k in mergeable and k in self:
                self[k].update(v)
            else:
                self[k] = v

    def get_detdb(self):
        filename = self['detdb'].format(depots=self['depots'])
        return sotodlib.metadata.DetDB.from_file(filename)

    def get_obsfiledb(self):
        import sotoddb
        filename = self['obsfiledb'].format(depots=self['depots'])
        return sotodlib.metadata.ObsFileDB.from_file(filename)


def _read_cfg(filename=None, envvar=None, default=None):
    """Load a YAML file.  If filename is None, use the filename specified
    in the environment variable called envvar.  If that is not defined
    or decodes to None or an empty string, use the filename specified
    in default.

    Returns (ok, full_path, data) where ok is a boolean indicating
    whether the file at full_path was found on the file-system,
    full_path is the full path to the resolved filename (or None if
    not resolved), and data is the OrderedDict containing the data (or
    {} if not decoded).

    """
    if filename is None and envvar is not None:
        filename = os.getenv(envvar)
        if filename is None or filename == '':
            filename = None
    if filename is None and default is not None:
        filename = default
    if filename is None:
        return False, None, odict()
    filename = os.path.abspath(filename)
    if not os.path.exists(filename):
        return False, filename, odict()
    return True, filename, yaml.safe_load(open(filename, 'r'))
