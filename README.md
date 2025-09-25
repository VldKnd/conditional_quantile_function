### Vector Quantile Estimation
We propose a framework for conditional vector quantile regression (CVQR) that
combines neural optimal transport with amortized optimization, and apply it to
multivariate conformal prediction. Classical quantile regression does not extend
naturally to multivariate responses, while existing approaches often ignore the
geometry of joint distributions. Our method parameterizes the conditional vector
quantile function as the gradient of a convex potential implemented by an input-
convex neural network, ensuring monotonicity and uniform ranks. To reduce the
cost of solving high-dimensional variational problems, we introduce amortized
optimization of the dual potentials, yielding efficient training and faster inference.
We then exploit the induced multivariate ranks for conformal prediction, con-
structing distribution-free predictive regions with finite-sample validity. Unlike
coordinatewise methods, our approach adapts to the geometry of the conditional
distribution, producing tighter and more informative regions. Experiments on
benchmark datasets show improved coverage–efficiency trade-offs compared to
baselines, highlighting the benefits of integrating neural optimal transport with
conformal prediction.

### Codebase
This codebase contains implimentation, benchmarking and experiments related to vector quantile regression paper.

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
* notebooks/ - Contain different implementations of approaches to quantile regression and their conformalization.
* src/ - Folder with main code snippets used in notebooks and experiments. Contains both conformal and optimal transport implementations.
* scripts/ - Folder with utility scripts to run processes in different infrastructure environments.
* poc/ - Notebooks with different experiments with Optimal Transport approach to quantile regression. Code in that folder is written to quickly check the idea and should not be treated as production ready.

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