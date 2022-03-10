class ArchivePolicy:
    """Storage policy assistance.  Helps to determine the HDF5
    filename and dataset name for a result.

    Make me better!

    """
    @staticmethod
    def from_params(params):
        if params['type'] == 'simple':
            return ArchivePolicy(**params)

    def __init__(self, **kwargs):
        self.filename = kwargs['filename']

    def get_dest(self, product_id):
        return self.filename, product_id
