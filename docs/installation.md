# Installation 


## On your local machine
Clone the repository and install fluxie:
```
git clone https://github.com/openghg/fluxie.git
cd fluxie
pip install -e .
```

## At ICOS Jupyter Hub
Create and activate virtual environment:
```
python -m venv fluxie-env         
source fluxie-env/bin/activate
```
Install IPython kernel package for Jupyter into the current environment and register the current environment as a new kernel:
```
pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name fluxie-env --display-name "fluxie-env"
```
And finally install fluxie:
```
git clone https://github.com/openghg/fluxie.git
cd fluxie
pip install -e .
```




## Installing the docs

The doc of fluxie is built using [mkdocs](https://www.mkdocs.org/). You can install it using pip:

```
pip install mkdocs mkdocs-material mkdocstrings[python]
```

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

### Commands

* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

### Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
