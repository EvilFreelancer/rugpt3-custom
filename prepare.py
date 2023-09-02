import pandas as pd
import os
from process_data import process_data

source_file = "data/output.csv"
file_extension = os.path.splitext(source_file)[1]

if file_extension == ".csv":
    data = pd.read_csv(source_file)
    data = data['text'].dropna()
elif file_extension == ".txt":
    with open(source_file, "r") as file:
        lines = file.readlines()
    data = pd.DataFrame(lines, columns=["text"])
else:
    raise ValueError("Unsupported file format. Please use .csv or .txt files.")

process_data(dataset=data)
