import numpy as np
import pandas as pd
import joblib
import argparse
import logging
from sklearn.linear_model import BayesianRidge
from oadap.prediction.utils import chauvenet

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fit and save a Bayesian Ridge Regression model for TAlk."
    )
    parser.add_argument(
        "--csv_file",
        default="data/MWRA/MWRA.csv",
        type=str,
        help="Path to raw MWRA CSV file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="checkpoints/TAlk_regression/model.pkl",
        type=str,
        help="Path to save the model checkpoint.",
    )
    return parser.parse_args()


args = parse_arguments()

# Parameters from the original notebook
ph_var = "pH ()"  # ph out
TA_var = "TA in (mmol/kgSW)"
sal_var = "SAL (PSU)"
temp_var = "TEMP (C)"
depth_var = "DEPTH (m)"
station_var = "STAT_ID"
aragonite_var = "WAr out"
TCO2_var = "TCO2 in (mmol/kgSW)"
ox_var = "DISS_OXYGEN (mg/L)"
fluor_var = "FLUORESCENCE (ug/L)"
date_var = "PROF_DATE_TIME_LOCAL"

# Stations to include
stations = [
    "F13",
    "F06",
    "F10",
    "F15",
    "N07",
    "N18",
    "N21",
    "F22",
    "N04",
    "N01",
    "F23",
    "F17",
    "N10",
    "F05",
]
test_station = "F06"

# Load the dataset
df = pd.read_csv(args.csv_file)

# Handle missing data and rename columns
df.loc[df["VAL_QUAL"] == -1, ph_var] = np.NaN
df = df[
    [
        station_var,
        "LATITUDE",
        "LONGITUDE",
        depth_var,
        temp_var,
        sal_var,
        ox_var,
        fluor_var,
        TA_var,
        TCO2_var,
        ph_var,
        aragonite_var,
        date_var,
    ]
]
df.columns = [
    "Station",
    "Latitude",
    "Longitude",
    "Depth",
    "Temperature",
    "Salinity",
    "Oxygen",
    "Fluorescence",
    "TA",
    "TCO2",
    "pH",
    "Ar",
    "Date",
]

# Convert 'Date' to datetime and filter out specific stations
df["Date"] = pd.to_datetime(df["Date"])
df = df[~df["Station"].isin(["HAR", "NFAL", "POC"])]

# Prepare the data
df_model = df.copy()
df_model = df_model[df_model["Station"].isin(stations)]
df_model = df_model[~df_model["Salinity"].isna()]
df_model = df_model[~df_model["TA"].isna()]
df_model = df_model.loc[chauvenet(df_model["TA"])]
df_model = df_model.loc[chauvenet(df_model["Salinity"])]

# Split into training and testing sets
df_train = df_model[df_model["Station"] != test_station]
df_test = df_model[df_model["Station"] == test_station]

X_train = df_train[["Salinity"]].values
y_train = df_train["TA"].values
X_test = df_test[["Salinity"]].values
y_test = df_test["TA"].values

# Train the Bayesian Ridge Regression model
model = BayesianRidge().fit(X_train, y_train)

# Save the model checkpoint
joblib.dump(model, args.checkpoint_path)
logging.info(f"Model checkpoint saved to {args.checkpoint_path}")