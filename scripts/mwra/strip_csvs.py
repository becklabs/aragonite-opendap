import os

import pandas as pd

input_dir = "data/MWRA/"
output_dir = "data_pub/MWRA/"
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
df = pd.read_csv(input_dir + "MWRA_clean.csv")

v = "TCO2"  # Target variable
regressors = ["Temperature", "Salinity", "Surface Chlorophyll"]
required_columns = [v, "Date", "Station"] + regressors + ["Latitude", "Longitude", "Depth"]

# Remove all columns that are not required
df = df[required_columns]
df.to_csv(output_dir + "MWRA_clean.csv", index=False)


df = pd.read_csv(input_dir + "MWRA.csv")

required_columns = ["TA in (mmol/kgSW)", "SAL (PSU)", "STAT_ID"]

# Remove all columns that are not required
df = df[required_columns]

df.to_csv(output_dir + "MWRA.csv", index=False)
