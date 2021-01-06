# Source Code

The real magic happens in this directory.

## Building
This project uses **conda**, so make sure to have it installed first.
Then, you can create yourself an environment with the necessary dependencies:

```bash
git clone git@git.rwth-aachen.de:chan.yong.lee/conformance-checking.git
cd conformance-checking/Code
conda env create -f conda-env.yml
```

From now on, you only need to run the following command before development:

```
conda activate conformance-checking
```


## Tests
The test suite covers many aspects:

- documentation tests via *doctest*, e.g. example python shell code (with `>>>`) are tested automatically.
- unit and integration tests via *pytest*, see `Code/tests/` for examples. Test files and functions have to start with `test_`.
- *flake8* is a linter catching common mistakes in python code
- *black* is a code formatter

The tests are run via *tox* on multiple python versions to ensure compatibility.
Executing the test suite should be very simple, just run:

```bash
tox
```

This can take a while the first time, but *tox* uses caching to speed up following executions.
Additionally, tests are run automatically for every commit pushed to the repository via GitLab Pipelines.
To check for test coverage, you can run:

```bash
pytest --cov --cov-report=html
```

This will create a HTML report.
Just open `htmlcov/index.html` with your browser.


## Dependency Management
Our dependencies are listed in `conda-env.yml` and `tox.ini`.
*Both files have to be kept in sync!*
If you change a dependency, please add it to both files.
To re-create the development environment, execute the following steps in the `Code` folder:

```bash
conda activate
conda env remove conformance-checking
conda env create -f conda-env.yml
conda activate conformance-checking
tox -r
```

The additional `tox -r` re-runs the test suite on the changed dependencies to ensure compatibility.
The `-r` tells *tox* to re-create the testing environments.
This can take some time.
