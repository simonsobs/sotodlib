.. code-block:: console

    usage: so_hardware_plot [options] (use --help for details)

    This program reads a hardware model and plots the detectors. Note that you
    should pre-select detectors before passing a hardware model to this function.
    See so_hardware_trim.

    optional arguments:
      -h, --help           show this help message and exit
      --hardware HARDWARE  Input hardware file
      --out OUT            Name of the output PDF file.
      --width WIDTH        The width of the plot in degrees.
      --height HEIGHT      The height of the plot in degrees.
      --labels             Add pixel and polarization labels to the plot.
