.. code-block:: console

    usage: so_data_export [options] (use --help for details)

    Copy observation data and metadata to the local system

    options:
        -h, --help              show this help message and exit
        --local_data LOCAL_DATA
                                The local data directory. Assumes <telescope> subdir.
        --local_metadata LOCAL_METADATA
                                The local metadata directory. Assumes <telescope>/manifests subdir.
        --local_context LOCAL_CONTEXT
                                The local context file to create
        --remote_data REMOTE_DATA
                                The remote data directory. Assumes <telescope> subdir.
        --remote_metadata REMOTE_METADATA
                                The remote metadata directory. Assumes <telescope>/manifests subdir.
        --remote_context REMOTE_CONTEXT
                                The context file path on the remote system
        --remote_host REMOTE_HOST
                                The address of the remote host (user@remote.domain.name)
        --obs_file OBS_FILE     A text file with the obs_ids to sync
        --ssh_control SSH_CONTROL
                                The file for the persistent ssh control socket
        --localize              If enabled, localize databases
