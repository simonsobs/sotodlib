# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import argparse
from functools import wraps

import toast
import toast.mpi


def workflow_timer(f):
    """Decorator for timing workflow functions.

    This requires that the workflow function conform to the standard
    function signature.

    """
    fname = f.__qualname__

    @wraps(f)
    def df(job, otherargs, runargs, *args):
        log = toast.utils.Logger.get()
        timer = toast.timing.Timer()
        timer.start()

        data = None
        comm = None
        for arg in args:
            if isinstance(arg, toast.Data):
                data = arg
                comm = data.comm.comm_world
            elif isinstance(arg, toast.mpi.MPI_Comm):
                comm = arg

        log.info_rank(f"Running {fname}...", comm=comm)

        result = f(job, otherargs, runargs, *args)

        log.info_rank(f"Finished {fname} in", comm=comm, timer=timer)
        if data is not None:
            job.operators.mem_count.prefix = f"After {fname}"
            job.operators.mem_count.apply(data)
        return result

    return df


def setup_job(
    parser=None,
    operators=list(),
    templates=list(),
    opts=None,
    config_files=None,
    argparse_opts=None,
    verbose=False,
):
    """Generate configuration for a workflow and parse options.

    Given a list of operators and templates to be used in a workflow, build the
    default configuration.  Then parse any option overrides from the commandline
    OR from the combination of argparse option dictionary and config files.  If
    an existing ArgumentParser is given, then it is used.  Otherwise a new empty
    parser is created.

    This function is designed to work from a script running in a batch job as well
    as interactively from a notebook.  The final configuration can optionally be
    written out for reference.

    For interactive use or ad-hoc scripts, the `opts` dictionary can be passed with
    options matching typical commandline syntax without the leading dashes.  A list
    of config files to load can also be supplied with `config_files`.

    For non-interactive scripts that need to parse commandline arguments, leave
    `opts` and `config_files` at None.  Or the commandline can be replaced with a
    list of argparse-style options in `argparse_opts`.

    Args:
        parser (ArgumentParser):  The input parser to modify.  If None, a new one
            is created.
        operators (list):  The list of operator instances to configure.
        templates (list):  The list of template instances to configure.
        opts (dict):  The dictionary of interactive config options.
        config_files (list):  A list of configuration files to load.
        argparse_opts (list):  When running in batch mode, an argparse list of
            command line arguments to use instead of parsing sys.argv.
        verbose (bool):  If True, dump out the final config to stdout.

    Returns:
        (tuple):  The (job namespace, config dictionary, other args, runtime args)
            from the result of parsing opts + config_files or command line and then
            configuring all operators and templates.

    """
    # We set up the utility memory tracking operator here, so it is always available
    # to any of the workflows.
    operators.append(toast.ops.MemoryCounter(name="mem_count", enabled=False))

    # Check for consistent arguments
    interactive = opts is not None or config_files is not None
    if interactive and argparse_opts is not None:
        msg = "Either opts/config_files should be used OR argparse_opts"
        raise ValueError(msg)

    if interactive:
        # Build the equivalent list of argparse options
        argparse_opts = list()
        if opts is not None:
            for k, v in opts.items():
                optkey = f"--{k}"
                if isinstance(v, bool):
                    # This is just a switch
                    if v:
                        argparse_opts.extend([optkey])
                    else:
                        # Invert
                        kparts = k.split(".")
                        kparts[-1] = f"no_{kparts[-1]}"
                        argparse_opts.extend([f"--{'.'.join(kparts)}"])
                else:
                    argparse_opts.extend([optkey, str(v)])
        if config_files is not None:
            for conf in config_files:
                argparse_opts.extend(["--config", conf])

    if parser is None:
        # Create a parser
        parser = argparse.ArgumentParser(description="Simons Obs TOAST Workflow")

    parser.add_argument(
        "--full_pointing",
        required=False,
        default=False,
        action="store_true",
        help="Save detector pointing rather than regenerating to save memory.",
    )
    parser.add_argument(
        "--dry_run",
        required=False,
        default=False,
        action="store_true",
        help="Exit after setting up job options.",
    )

    # The "config" is the internal representation of all config parameters
    # merged from defaults, config files, and command line arguments.  Otherargs
    # are anything else parsed from opts / CLI.  Runargs contains runtime parameters
    # like procs per node, groupsize, etc.
    config, otherargs, runargs = toast.config.parse_config(
        parser,
        operators=operators,
        templates=templates,
        prefix="",
        opts=argparse_opts,
    )

    # Instantiate operators and templates
    job = toast.config.create_from_config(config)

    if verbose:
        out = ""
        for objtype, objlist in zip(
            ["Operators", "Templates"],
            [job.operators, job.templates],
        ):
            out += f"{objtype}:\n"
            for objname, obj in vars(objlist).items():
                out += f"  {objname}:\n"
                for trait_name, trait in obj.traits().items():
                    if trait_name == "API":
                        continue
                    if trait_name == "kernel_implementation":
                        continue
                    if trait_name == "name":
                        continue
                    out += f"    {trait_name} = {trait.get(obj)} # {trait.help}\n"
        print(out)

    return job, config, otherargs, runargs
