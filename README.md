# NSP2-Alpha-Ray

## Documentation
https://m-serrano-altena.github.io/NSP2-Alpha-Ray/

## Add repository
To add this repository:
Go to a folder that you want to add a folder with all files in this repository.
In the terminal of vscode:
```
git clone https://github.com/M-Serrano-Altena/NSP2-Alpha-Bron.git
```

Then create the 'alpha' environment:
```
conda create -n alpha python=3.10
conda activate alpha
```

In the 'alpha' environment:
```
poetry install
```

## Install poetry
If poetry isn't installed:
Open anaconda prompt:
```
conda create -n pipx python
conda activate pipx
python -m pip install --user pipx
python -m pipx ensurepath
pipx install poetry
```

