# Installing Pyenv

This is a quick summary of how to get started with Pyenv and Python version/environment management. See the [Pyenv Github](https://github.com/pyenv/pyenv) for more details.

Other environment managers like `conda` will also work, if preferred.

### Install dependencies:
```
sudo apt-get update; sudo apt-get install --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

### Install Pyenv
```
curl https://pyenv.run | bash
```

### Set up your `~/.bashrc`

Check that the following lines below are included in the file. If not, paste them in.
```
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

## Activate your environment

This repo was tested using Python `3.10.8`, however, Pyenv supports most Python versions (run `pyenv install -l | grep '^  [0-9]'` to see what is available to install). If another Python version is desired, replace `3.10.8` in the code below with your version.

If you have not previously installed Python with Pyenv, run
```
pyenv install 3.10.8
```
Then, the following lines will create the environment and activate it for the current shell session.
```
pyenv virtualenv 3.10.8 cbfpy
pyenv shell cbfpy
```
!!! tip
    Use `pyenv local cbfpy` in the top-level `cbfpy` directory to automatically switch to this environment when working in this directory.

