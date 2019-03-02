.. code-block:: console

    usage: so_hardware_trim [options] (use --help for details)

    This program reads a hardware model from disk, selects a subset of detectors,
    and writes the new model out.

    optional arguments:
      -h, --help            show this help message and exit
      --hardware HARDWARE   Input hardware file
      --out OUT             Name (without extensions) of the output hardware file
      --plain               Write plain text (without gzip compression)
      --overwrite           Overwrite any existing output file.
      --telescopes TELESCOPES
                            Select only detectors on these telescope (LAT, SAT0,
                            SAT1, etc) . This should be either a regex string or a
                            comma-separated list of names.
      --tubes TUBES         Select only detectors on these tubes. This should be
                            either a regex string or a comma-separated list of
                            names.
      --match [MATCH [MATCH ...]]
                            Specify one or more detector match criteria. Each
                            match should have the format '<property>:<regex or
                            list>'. The regex expression should be valid to pass
                            to the 're' module. If passing a list, this should be
                            comma-separated. For example, --match 'band:MF.*'
                            'wafer:25,26' 'pol:A'
