import os


def process_data(dataset, prefix=None, suffix=None, frac=0.7, seed=42, save_path="data"):
    train = dataset.sample(frac=frac, random_state=seed).copy()
    valid = dataset[~dataset.index.isin(train.index)].copy()

    train = [str(t).strip() for t in train]
    valid = [str(v).strip() for v in valid]

    if prefix is not None:
        train = [prefix + str(t) for t in train]
        valid = [prefix + str(v) for v in valid]

    if suffix is not None:
        train = [str(t) + suffix for t in train]
        valid = [str(v) + suffix for v in valid]

    with open(os.path.join(save_path, "train.txt"), "w") as file:
        file.write("\n".join(train))

    with open(os.path.join(save_path, "valid.txt"), "w") as file:
        file.write("\n".join(valid))
