import pandas as pd
import os


dataframe = pd.read_parquet("pet.parquet")
print(dataframe.columns)
dataframe = dataframe[["document name", "tokens"]]
dataframe = dataframe.to_dict("records")

for doc in dataframe:
    file_name = os.path.join("pet", doc["document name"])
    F = open(file_name, "w")
    stru = " ".join(doc["tokens"])
    F.write(stru.replace(" . ", ".\n\n"))
    F.close()
