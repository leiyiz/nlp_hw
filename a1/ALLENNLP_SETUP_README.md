## Installation

AllenNLP requires Python 3.6.1 or later. The preferred way to install AllenNLP is via `pip`.  

To set up for A1, run `pip install git+git://github.com/allenai/allennlp.git@088f0bb` [[1]](#shaexplanation) in your Python environment and you're good to go!

If you need pointers on setting up an appropriate Python environment or would like to install AllenNLP using a different method, see below.

Windows is currently not officially supported, although the AllenNLP team tries to fix issues when they are easily addressed.

### Installing via pip

#### Setting up a virtual environment

[Conda](https://conda.io/) can be used set up a virtual environment with the
version of Python required for AllenNLP. If you already have a Python 3.6 or 3.7
environment you want to use, you can skip to the 'installing via pip' section.

1.  [Download and install Conda](https://conda.io/docs/download.html).

2.  Create a Conda environment with Python 3.6

    ```bash
    conda create -n allennlp python=3.6
    ```

3.  Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use AllenNLP.

    ```bash
    source activate allennlp
    ```

#### Installing the library and dependencies

Installing the library and dependencies is simple using `pip`.

   ```bash
   pip install git+git://github.com/allenai/allennlp.git@088f0bb
   ```

That's it! You're now ready to build and train AllenNLP models.
AllenNLP installs a script when you install the python package, meaning you can run allennlp commands just by typing `allennlp` into a terminal.

---

<a name="shaexplanation">[1]</a> Why the Git SHA? Your course staff procrastinated in setting up this assignment, so some of the code necessary for A1 was [only recently merged](https://github.com/allenai/allennlp/pull/2264) into `allennlp`, and thus hasn't made it into a `pip` release yet.
