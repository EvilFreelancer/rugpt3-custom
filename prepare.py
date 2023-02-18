import pandas as pd

data = pd.read_csv("data/output.csv")
data.head()

train = data.sample(frac=0.7, random_state=42).copy()
val = data[~data.index.isin(train.index)].copy()

train = [train.text.iloc[idx] for idx in range(len(train)) if type(train.text.iloc[idx]) != float]
valid = [val.text.iloc[idx] for idx in range(len(val)) if type(val.text.iloc[idx]) != float]

len(train), len(valid)

with open("data/train.txt", "w") as file:
    file.write("\n".join(train))

with open("data/valid.txt", "w") as file:
    file.write("\n".join(valid))
