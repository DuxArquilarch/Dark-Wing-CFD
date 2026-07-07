# Dark Wing CFD

Requires Python 3.9+.

## Dependencies

numpy, scipy, matplotlib, pillow, numba (optional, only speeds up the solver)

Tkinter is also required (comes with Python on Windows, separate package on Linux).

## Windows

Install Python from https://www.python.org/downloads/windows/
(check "Add python.exe to PATH" during install).

Open cmd:

```cmd
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib pillow numba
```

Run:

```cmd
cd C:\path\to\Dark Wing CFD
python main.py
```

## Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-tk
python3 -m pip install --upgrade pip
python3 -m pip install numpy scipy matplotlib pillow numba
```

Run:

```bash
cd /path/to/Dark\ Wing\ CFD
python3 main.py
```

## Linux (Fedora)

```bash
sudo dnf install -y python3 python3-pip python3-tkinter
python3 -m pip install numpy scipy matplotlib pillow numba
```

## Linux (Arch)

```bash
sudo pacman -S --needed python python-pip tk
python -m pip install numpy scipy matplotlib pillow numba
```

IPT 120C && IDK.ipt or any stl can be loaded in the program using the import stl configuration

