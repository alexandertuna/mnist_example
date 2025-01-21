from pathlib import Path
from typing import Tuple
import numpy as np
np.set_printoptions(linewidth=100000)

def main() -> None:
    this_path = Path(__file__)
    data_path = this_path.parent / "mnist_train.csv"
    labels, images = parse(data_path)
    print(images[0])

def parse(fpath: Path) -> Tuple[np.array, np.array]:
    data = np.loadtxt(fpath, delimiter=",")
    labels = data[:, 0]
    images = data[:, 1:]
    return labels, images.reshape([-1, 28, 28])

if __name__ == "__main__":
    main()
