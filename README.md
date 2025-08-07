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

