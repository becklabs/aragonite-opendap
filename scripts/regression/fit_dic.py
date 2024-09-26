import os
import pandas as pd
import GPy
import joblib
import argparse
import logging

from oadap.prediction.utils import chauvenet

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fit and save a Gaussian Process Regression model for TC02."
    )
    parser.add_argument(
        "--csv_file",
        default="data/MWRA/MWRA_clean.csv",
        type=str,
        help="Path to cleaned MWRA CSV file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="checkpoints/DIC_regression/model.pkl",
        type=str,
        help="Path to save the model checkpoint.",
    )
    return parser.parse_args()


args = parse_arguments()

# Parameters from the original script
v = "TCO2"  # Target variable
test_s = "F06"  # Station to use as test set
linear = False  # Use Gaussian Process Regression
optimize = False  # Do not optimize hyperparameters
regressors = ["Temperature", "Salinity", "Surface Chlorophyll"]
units = {
    "Temperature": "$^\circ$C",
    "Salinity": "PSU",
    "Surface Chlorophyll": "mg/m$^-3$",
    "TCO2": "$\mu$mol/kg",
}

# Load the dataset
df = pd.read_csv(args.csv_file)

# Ensure required columns are present
required_columns = [v, "Date", "Station"] + regressors
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in the input data.")

# Convert 'Date' to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Copy and preprocess the data
data = df.copy()
for r in regressors:
    data = data[~data[r].isna()]
    data = data.loc[chauvenet(data[r])]
data = data[~data[v].isna()]

checkpoint = {}

# Define seasons
seasons = {0: "October - March", 1: "April - June", 2: "July - September"}
seasons_idx = [[10, 11, 2, 3], [4, 5, 6], [7, 8, 9]]
ns = len(seasons_idx)

# Loop over each season to train models
for j in range(ns):
    season_name = seasons[j]
    months = seasons_idx[j]
    df_d = data[data["Date"].dt.month.isin(months)].copy()
    df_d = df_d[~df_d[v].isna()]

    if df_d.empty:
        logging.warning(f"No data available for season {season_name}. Skipping.")
        continue

    # Standardize regressors and target variable
    Xm = df_d[regressors].mean(axis=0).values
    Xs = df_d[regressors].std(axis=0).values
    df_d[regressors] = (df_d[regressors] - Xm) / Xs  # type: ignore

    ym = df_d[v].mean()
    ys = df_d[v].std()
    df_d[v] = (df_d[v] - ym) / ys

    # Split into training and testing sets
    df_train = df_d[df_d["Station"] != test_s]
    df_test = df_d[df_d["Station"] == test_s]

    X_train = df_train[regressors].values
    y_train = df_train[v].values.reshape(-1, 1)  # type: ignore

    # Define and train the model
    if linear:
        from sklearn.linear_model import BayesianRidge

        model = BayesianRidge().fit(X_train, y_train.ravel())
    else:
        kernel = GPy.kern.RBF(input_dim=X_train.shape[1], ARD=True)
        model = GPy.models.GPRegression(X_train, y_train, kernel)

        if optimize:
            model.optimize()
        else:
            # Fix hyperparameters if not optimizing
            if len(regressors) == 4:
                lengthscales = [1, 0.5, 1, 1]
            elif len(regressors) == 3:
                lengthscales = [0.5, 1, 1]
            else:
                lengthscales = [1.0] * X_train.shape[1]
            model.rbf.lengthscale.fix(lengthscales)
            model.rbf.variance.fix(1.3)
            model.Gaussian_noise.variance.fix(0.05)

    # Save the model and normalization parameters
    checkpoint[season_name] = {
        "model": model,
        "Xm": Xm,
        "Xs": Xs,
        "ym": ym,
        "ys": ys,
    }

# Save the checkpoint
if not os.path.exists(os.path.dirname(args.checkpoint_path)):
    os.makedirs(os.path.dirname(args.checkpoint_path))
joblib.dump(checkpoint, args.checkpoint_path)
logging.info(f"Model checkpoint saved to {args.checkpoint_path}")
