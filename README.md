REGen
=====

Prerequisites:
--------------
* SymPy
* Jupyter
* jupyter_contrib_nbextensions

It is recommended to install Anaconda (based on Python 3) and create
an environment for the REGen project:

    $ conda create --name regen
    $ . activate regen
    (regen) $ conda install sympy
    (regen) $ conda install jupyter
    (regen) $ conda install jupyter_contrib_nbextensions
    (regen) $ jupyter nbextension install --py latex_envs
    (regen) $ jupyter nbextension enable --py latex_envs
