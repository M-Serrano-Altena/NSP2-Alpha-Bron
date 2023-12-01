# NSP2-Alpha-Bron

To add this repository:
Go to a folder that you want to add a folder with all files in this repository.
In the terminal of vscode:
```
git clone https://github.com/M-Serrano-Altena/NSP2-Alpha-Bron.git
```

Then create the alfa environment:
```
conda create -n alfa python=3.10
conda activate alfa
```

In the environment:
```
poetry install
```

If poetry isn't installed:
Open anaconda prompt:
```
conda create -n pipx python
conda activate pipx
python -m pip install --user pipx
python -m pipx ensurepath
pipx install poetry
```

