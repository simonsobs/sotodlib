.. py:module::sotodlib.site_pipeline.utils.profiler

Profiling Workflows
===================

Many of the critical workflows in the site-pipeline (e.g. data packaging) can
be difficult to understand in isolation due to their requirements on external
resources like databases. As such, we have developed 'live' profiling tools
that allow flows to be performed with a profiler enabled. We provide two core
functions:

.. code-block:: python
   
   from sotodlib.site_pipeline.utils.profiler import profile, add_profile_args

   def other_function(x):
       return x**2

   @profile("example")
   def main(value):
       value *= 7
       ysum = 0
       for x in range(value):
           y = other_function(x)
           ysum += y

       print(f"Value: {value}")

       return ysum


   if __name__ == "__main__":
       import argparse as ap

       parser = ap.ArgumentParser()
       parser.add_argument("value", type=int)
       add_profile_args(parser)

       args = parser.parse_args()
       main(**vars(args))


These functions allow you to add three function and CLI arguments to your ``main``
functions: 

* ``profile``: a boolean argument that tells the script to enable profiling if provided.
  This defaults to false.
* ``profile_type``: a string argument having one of the following three values:

   * ``html``: Outputs a HTML-viewable webpage rendered by ``pyinstrument``
   * ``txt``: Outputs a text-rendered profile from ``pyinstrument``
   * ``lprof``: Outputs a text-rendered profile from ``lprof``

* ``profile_output``: The directory to save the profile files into. Defaults to
  ``/data/data-package/profiles``.

We default to the ``pyinstrument`` profiler with ``html`` output. Profiles are
saved with filenames as given by the first argument to ``@profile`` and a timestamp,
to avoid over-writing, e.g. ``example_2026-06-22T131816.html``.

Types of Profiles
-----------------

We provide two types of profile, from two underlying profiling libraries:

* A call stack summary (stochastic) profile from ``pyinstrument`` (for ``html`` 
  and ``txt`` outputs), which has a small performance impact. This tells you how
  long was spent in each function call.
* A full tracing line profile from ``line_profiler`` (for ``lprof`` outputs),
  which may have a significant performance impact. This tells you how many times
  each line in the file was visited and how much of the program's runtime was
  spent there. We automatically include all files inside ``sotodlib`` itself.

Generally, we recommend starting with the default ``pyinstrument -> html`` profile.
If more fine-grained information is required, the ``line_profiler`` may be worth
trying.