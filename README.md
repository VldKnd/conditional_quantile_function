### Conditional Quantile Estimation
Codebase containing implimentation, benchmarking and experiments related to vector quantile regression.

### Installation
You can manage the Python dependencies for this project using uv or pip.

### Project Dependencies
uv is an extremely fast Python package installer and resolver. If you'd like to use it, we recommned using it. It is an almost drop-in replacement of pip. Configuration file should be compatible with pip, so if you want you can use pip instead.
```bash
## Using uv:

# Optional: Installing uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Installing dependencies
uv sync
```
Or
```bash
## Using pip:

# Create a virtual environment:
python -m venv .venv
source .venv/bin/activate

# Install the dependencies using pip:
pip install -r requirements.txt
```

### Project structure
* notebooks/ - Contain different implementations of approaches to quantile regression.
* src/ - Folder with main code snippets used in notebooks and experiments.
* poc/ - Notebooks with different experiments with Optimal Transport approach to quantile regression.

### Commit messages:
```
Format: 
* <type>(<scope>): <subject>
* <type>: <subject> 
```

Example
```
feat(my_feature_name): adds new cool feature
^--^  ^------------^   ^-------------------^
|     |                 |
|     |                 +-> Description in present tense
|     |
|     +-> Optional scope of the commit
|
+-------> Type: chore, docs, feat, fix, refactor, style, or test.
```

More Examples:
```
- feat: (new feature for the user, not a new feature for build script)
- fix(optimal_transport): (bug fix for the user, not a fix to a build script)
- docs(conformal): (changes to the documentation)
- style: (formatting, missing semi colons, etc; no production code change)
- refactor(conformal): (refactoring production code, eg. renaming a variable)
- test: (adding missing tests, refactoring tests; no production code change)
- chore: (updating grunt tasks etc; no production code change)
```