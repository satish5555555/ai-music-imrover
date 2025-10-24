import os
from pathlib import Path
def makedirs(path): Path(path).mkdir(parents=True, exist_ok=True)
