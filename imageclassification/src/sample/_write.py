from ._types import Dataset


def write_dataset(dataset: Dataset, filename: str):
    """
    TODO
    """
    with open(filename, "w") as file:
        for f in dataset.keys():
            file.write(f + "\n")
