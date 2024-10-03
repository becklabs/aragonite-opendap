import pandas as pd
import joblib
import argparse
import logging
from sklearn.linear_model import BayesianRidge
from oadap.prediction.utils import chauvenet

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fit and save a Bayesian Ridge Regression model for TA."
    )
    parser.add_argument(
        "--csv_file",
        default="data/MWRA/MWRA.csv",
        type=str,
        help="Path to the raw MWRA CSV file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="checkpoints/TAlk_regression/model.pkl",
        type=str,
        help="Path to save the model checkpoint.",
    )
    return parser.parse_args()

args = parse_arguments()

# Essential variables
TA_var = "TA in (mmol/kgSW)"
sal_var = "SAL (PSU)"
station_var = "STAT_ID"

# Stations to include
stations = [
    "F13", "F06", "F10", "F15", "N07", "N18",
    "N21", "F22", "N04", "N01", "F23", "F17",
    "N10", "F05",
]
test_station = "F06"

# Load the dataset
df = pd.read_csv(args.csv_file)

# Select and rename the essential columns
df = df[[station_var, sal_var, TA_var]]
df.columns = ["Station", "Salinity", "TA"]

# Filter out specific stations
df = df[~df["Station"].isin(["HAR", "NFAL", "POC"])]

# Prepare the data
df_model = df[df["Station"].isin(stations)]
df_model = df_model.dropna(subset=["Salinity", "TA"])
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
