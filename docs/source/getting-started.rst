.. _install:

Getting started
~~~~~~~~~~~~~~~

The best way to get started is to `install Anaconda or
Miniconda <https://conda.io/docs/user-guide/install/index.html>`__.
Once you've done that, fire up your favorite terminal emulator (PowerShell or
CMD on Windows, but we recommend CMD; iTerm2 or Terminal on Mac; lots of
choices if you're on Linux, but you knew that) and navigate to the base
directory of the repository (it should contain ``requirements.txt``).

On UNIX variants, type:

.. code:: bash

    $ conda env create -n hybridfactory -f requirements.txt
    Solving environment: done
    Downloading and Extracting Packages
    ...
    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done
    #
    # To activate this environment, use
    #
    #     $ conda activate hybridfactory
    #
    # To deactivate an active environment, use
    #
    #     $ conda deactivate

    $ conda activate hybridfactory

On Windows:

.. code:: shell

    $ conda env create -n hybridfactory
    Solving environment: done
    Downloading and Extracting Packages
    ...
    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done
    #
    # To activate this environment, use:
    # > activate hybridfactory
    #
    # To deactivate an active environment, use:
    # > deactivate
    #
    # * for power-users using bash, you must source
    #

    $ activate hybridfactory

and you should be good to go.
**Remember that**
``[conda] activate hybridfactory`` **every time you open up a new shell!**
