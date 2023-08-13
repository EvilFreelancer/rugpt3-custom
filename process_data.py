import pandas as pd
import re
import io


def process_data(prefix=None):
    data = pd.read_csv("data/output.csv")
    data.head()

    train = data.sample(frac=0.7, random_state=42).copy()
    val = data[~data.index.isin(train.index)].copy()

    train = [train.text.iloc[idx] for idx in range(len(train)) if type(train.text.iloc[idx]) != float]
    valid = [val.text.iloc[idx] for idx in range(len(val)) if type(val.text.iloc[idx]) != float]

    # Очистка текста от символов "(\ )?" и их замена на "\n<s>"
    train = [re.sub(r"                                                                    (\ )?", "\n" + prefix, t) for t in train]
    valid = [re.sub(r"                                                                    (\ )?", "\n" + prefix, v) for v in valid]

    train = [re.sub(r" \* \* \* ", "\n" + prefix, t) for t in train]
    valid = [re.sub(r" \* \* \* ", "\n" + prefix, v) for v in valid]

    train = [re.sub(r"(\ )+", " ", t) for t in train]
    valid = [re.sub(r"(\ )+", " ", v) for v in valid]

    train = [t.strip() for t in train]
    valid = [v.strip() for v in valid]

    # If prefix is set, then use it in array
    if prefix:
        train = [prefix + str(t) for t in train]
        valid = [prefix + str(v) for v in valid]

    len(train), len(valid)

    with open("data/train.txt", "w") as file:
        file.write("\n".join(train))

    with open("data/valid.txt", "w") as file:
        file.write("\n".join(valid))
