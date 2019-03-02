.. _dev:

Developer Guidelines
==============================

Here are some basic guidelines for contributing:

- This is a public repo.  Do not put any real or proprietary data files in this
  repo.  Use one of the private repos instead.  In general, large data files
  should not go in a git repo.

- If you are making changes larger than a small bugfix, make a branch for your
  changes and open a pull request when you are going to merge.  Even if you
  self-merge the PR, it notifies other developers about what is going on.

- When adding new code or modifying existing code, update the docstrings and
  (if needed) the sphinx documentation.  This code uses google style docstrings
  because it usually requires less typing and screen space to accomplish the
  same thing as other styles.

- When adding new code, try to add some kind of unit test too.

- If you find a problem or have a question about the best approach, open a
  github issue or ask on an appropriate open issue.

- The python code aims to follow PEP8 whenever possible.  Use a plugin for your
  editor of choice or a command line tool to easily check for problems.
  Developers should feel free to fix minor PEP8 issues if they find them.

- Scripts (a.k.a. "entry points") should have their main function defined in a
  file in the scripts directory (see existing tools there).  Then add an entry
  point to setup.py.  This allows running the script as part of the unit tests.
