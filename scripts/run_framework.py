import logging
import pandas as pd
import numpy as np
import os
import argparse  
from scipy.spatial import KDTree
from joblib import Memory
import xarray as xr  

from oadap.providers.opendap import GHRSSTL4, AQUAMODISCHLORL3
from oadap.providers.dataframe import MWRASalinity

from oadap.prediction.interpolation import (
    TimeSubsetInterpolator,
    KrigingInterpolator,
    ScaledKrigingInterpolator,
)

from oadap.prediction.utils import grid_to_swath
from oadap.utils import load_mat

from oadap.prediction.modules import (
    TCNModule,
    TAlkRegressionModule,
    DICRegressionModule,
    CO2SYSAragoniteModule,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process and save aragonite field data."
    )
    parser.add_argument(
        "--start", type=str, required=True, help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end", type=str, required=True, help="End date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="intermediate_cache/",
        help="Cache directory for joblib memory",
    )
    parser.add_argument(
        "--output_nc",
        type=str,
        default="aragonite_field.nc",
        help="Output NetCDF file path",
    )
    return parser.parse_args()


args = parse_arguments()
# Parse start and end dates
start = pd.Timestamp(args.start)
end = pd.Timestamp(args.end)
cache_dir = args.cache_dir
output_netcdf_path = args.output_nc

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the logging level to INFO
file_handler = logging.FileHandler('script.log')
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.propagate = False

# CONSTANTS
WINDOW_SIZE = 20

# VARIABLES
data_dir = "data/"
salinity_data_path = os.path.join(data_dir, "MWRA", "MWRA_clean.csv")

satellite_xy_path = os.path.join(data_dir, "EarthData", "sst_grid", "xy.npy")
neighbor_inds_path = os.path.join(data_dir, "EarthData", "sst_grid", "neighbors.npy")
satellite_xy = np.load(satellite_xy_path).reshape(-1, 2)
neighbor_inds = np.load(neighbor_inds_path).reshape(-1, 4)
satellite_xy_all = satellite_xy.copy()
satellite_xy = satellite_xy[np.all(neighbor_inds != -1, axis=1)]

chlor_ocean_mask_path = os.path.join(
    data_dir, "EarthData", "chlor_grid", "ocean_mask.npy"
)
chlor_ocean_mask = np.load(chlor_ocean_mask_path)

lon = load_mat(data_dir + "FVCOM/x.mat")
lat = load_mat(data_dir + "FVCOM/y.mat")
fvcom_xy = np.column_stack((lon, lat))

lat_min = np.min(fvcom_xy[:, 1])
lat_max = np.max(fvcom_xy[:, 1])
lon_min = np.min(fvcom_xy[:, 0])
lon_max = np.max(fvcom_xy[:, 0])

sal_tcn_config = "config/tcn/v0.yaml"
temp_tcn_config = "config/tcn/v0.yaml"

sal_tcn_checkpoint = "checkpoints/TCN/salinity/model_epoch_95.pth"
temp_tcn_checkpoint = "checkpoints/TCN/temperature/model_epoch_95.pth"

temp_fvcom_artifacts_dir = os.path.join(
    data_dir, "FVCOM", "preprocessed", "temperature", "all", "artifacts"
)
sal_fvcom_artifacts_dir = os.path.join(
    data_dir, "FVCOM", "preprocessed", "salinity", "all", "artifacts"
)

dic_regression_checkpoint = os.path.join("checkpoints", "DIC_regression", "model.pkl")
ta_regression_checkpoint = os.path.join("checkpoints", "TAlk_regression", "model.pkl")

fvcom_siglay_path = os.path.join(data_dir, "FVCOM", "siglay.mat")
fvcom_h_path = os.path.join(data_dir, "FVCOM", "h.mat")

# SET UP JOBLIB MEMORY FOR CACHING
INTERMEDIATE_DIR = cache_dir
memory = Memory(location=INTERMEDIATE_DIR, verbose=0)

####
data_start = start - pd.Timedelta(days=WINDOW_SIZE - 1)
data_end = end

# FETCHING DATA
salinity_provider = MWRASalinity(path=salinity_data_path)
temperature_provider = GHRSSTL4()
chlorophyll_provider = AQUAMODISCHLORL3()


@memory.cache
def fetch_salinity_data():
    logger.info("Fetching salinity data...")
    data = salinity_provider.subset(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        start=pd.Timestamp("2015-01-01"),  # Fetch all salinity data
        end=pd.Timestamp("2025-01-01"),
    )
    logger.info("Salinity data fetched.")
    return data


@memory.cache
def fetch_temperature_data():
    logger.info("Fetching temperature data...")
    data = temperature_provider.subset(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        start=data_start,
        end=data_end,
        workers=8,
        pbar=True,
    )
    logger.info("Temperature data fetched.")
    return data


@memory.cache
def fetch_chlorophyll_data():
    logger.info("Fetching chlorophyll data...")
    data = chlorophyll_provider.subset(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        start=start - pd.Timedelta(days=1),
        end=end + pd.Timedelta(days=1),
        workers=8,
        pbar=True,
    )
    logger.info("Chlorophyll data fetched.")
    return data


# Fetch data with caching
sal_raw, sal_xy, sal_time = fetch_salinity_data()
sst_raw, sst_xy, sst_time = fetch_temperature_data()
chlor_raw, chlor_xy, chlor_time = fetch_chlorophyll_data()

K_to_C = lambda K: K - 273.15
sst_raw = K_to_C(sst_raw)

# INTERPOLATION FUNCTIONS
@memory.cache
def compute_chlor_surface():
    logger.info("Computing chlorophyll surface interpolation...")
    chlor_interpolator = TimeSubsetInterpolator(
        n_days=1,
        start_date=start,
        end_date=end,
        interpolator_factory=lambda: KrigingInterpolator(variogram_model="exponential"),
    )

    chlor_raw_swath, chlor_xy_swath, chlor_time_swath = grid_to_swath(
        grid=chlor_xy[chlor_ocean_mask],
        values=chlor_raw[chlor_ocean_mask],
        time=chlor_time,
    )

    chlor_interpolator.fit(
        xy=chlor_xy_swath, values=np.log(chlor_raw_swath), time=chlor_time_swath
    )

    chlor_surface = np.exp(
        chlor_interpolator.predict(grid=satellite_xy, days=pd.date_range(start, end))
    )
    logger.info("Chlorophyll surface computed.")
    return chlor_surface


@memory.cache
def compute_salinity_surface():
    logger.info("Computing salinity surface interpolation...")
    sal_interpolator = ScaledKrigingInterpolator(variogram_model="linear")
    sal_interpolator.fit(xy=sal_xy, values=sal_raw, time=sal_time)

    salinity_surface = sal_interpolator.predict(
        grid=satellite_xy_all, days=pd.date_range(data_start, data_end)
    )
    logger.info("Salinity surface computed.")
    return salinity_surface


# RECONSTRUCT FIELDS
@memory.cache
def reconstruct_temp_field(sst_raw, sst_time):
    logger.info("Reconstructing temperature field...")
    fvcom_xy_temp = np.load(os.path.join(temp_fvcom_artifacts_dir, "xy.npy"))
    phi_temp = np.load(os.path.join(temp_fvcom_artifacts_dir, "phi.npy"))
    climatologies_temp = np.load(
        os.path.join(temp_fvcom_artifacts_dir, "climatology_days.npy")
    )
    fvcom_h = load_mat(fvcom_h_path)

    temperature_tcn = TCNModule(
        grid_xy=satellite_xy_all,
        grid_neighbor_inds=neighbor_inds,
        fvcom_xy=fvcom_xy_temp,
        fvcom_h=fvcom_h,
        fvcom_phi=phi_temp,
        fvcom_climatologies=climatologies_temp,
        config_path=temp_tcn_config,
        checkpoint_path=temp_tcn_checkpoint,
    )

    temp_field, output_grid, _ = temperature_tcn.predict(
        surface=sst_raw.reshape(-1, sst_raw.shape[-1]),
        time=sst_time,
    )
    logger.info("Temperature field reconstructed.")
    return temp_field


@memory.cache
def reconstruct_salinity_field(salinity_surface, sst_time):
    logger.info("Reconstructing salinity field...")
    fvcom_xy_sal = np.load(os.path.join(sal_fvcom_artifacts_dir, "xy.npy"))
    phi_sal = np.load(os.path.join(sal_fvcom_artifacts_dir, "phi.npy"))
    climatologies_sal = np.load(
        os.path.join(sal_fvcom_artifacts_dir, "climatology_days.npy")
    )
    fvcom_h = load_mat(fvcom_h_path)

    salinity_tcn = TCNModule(
        grid_xy=satellite_xy_all.reshape(-1, 2),
        grid_neighbor_inds=neighbor_inds,
        fvcom_xy=fvcom_xy_sal,
        fvcom_h=fvcom_h,
        fvcom_phi=phi_sal,
        fvcom_climatologies=climatologies_sal,
        config_path=sal_tcn_config,
        checkpoint_path=sal_tcn_checkpoint,
    )

    salinity_field, _, _ = salinity_tcn.predict(
        surface=salinity_surface,
        time=sst_time,
    )
    logger.info("Salinity field reconstructed.")
    return salinity_field


@memory.cache
def compute_dic_field(salinity_field, temp_field):
    logger.info("Computing DIC field...")
    dic_regressor = DICRegressionModule(checkpoint_path=dic_regression_checkpoint)

    dic_field = dic_regressor.predict(
        salinity_field=salinity_field,
        temperature_field=temp_field,
        chlorophyll_surface=chlor_surface,
        time=pd.date_range(start, end),
        pbar=True,
    )
    logger.info("DIC field computed.")
    return dic_field


@memory.cache
def compute_talk_field(salinity_field):
    logger.info("Computing TAlk field...")
    talk_regressor = TAlkRegressionModule(checkpoint_path=ta_regression_checkpoint)

    talk_field = talk_regressor.predict(
        salinity=salinity_field,
    )
    logger.info("TAlk field computed.")
    return talk_field


@memory.cache
def compute_aragonite_field(salinity_field, temp_field, dic_field, talk_field):
    logger.info("Computing aragonite field...")
    fvcom_h = load_mat(fvcom_h_path)
    fvcom_siglay = load_mat(fvcom_siglay_path)
    co2sys_module = CO2SYSAragoniteModule()

    tree = KDTree(fvcom_xy)
    dist, grid_idx_map = tree.query(satellite_xy)

    depth = fvcom_h[grid_idx_map] * -fvcom_siglay[grid_idx_map]

    aragonite_field = co2sys_module.predict(
        salinity_field=salinity_field,
        temperature_field=temp_field,
        dic_field=dic_field,
        talk_field=talk_field,
        depth=depth,
        pbar=True,
    )
    logger.info("Aragonite field computed.")
    return aragonite_field, depth


chlor_surface = compute_chlor_surface()

salinity_surface = compute_salinity_surface()

temp_field = reconstruct_temp_field(sst_raw, sst_time)
salinity_field = reconstruct_salinity_field(salinity_surface, sst_time)

dic_field = compute_dic_field(salinity_field, temp_field)
talk_field = compute_talk_field(salinity_field)

aragonite_field, depth = compute_aragonite_field(
    salinity_field, temp_field, dic_field, talk_field
)

logger.info("Saving aragonite field to NetCDF file...")

# Prepare the data for NetCDF
lon = satellite_xy[:, 0]
lat = satellite_xy[:, 1]
times = pd.date_range(start, end)  # n_times
n_locations = len(lon)
n_times = len(times)
n_depths = depth.shape[1]  # Assuming depth is of shape (n_locations, n_depths)

# Create an xarray Dataset
ds = xr.Dataset(
    {"aragonite": (("location", "time", "depth"), aragonite_field)},
    coords={
        "lon": ("location", lon),
        "lat": ("location", lat),
        "time": times,
        "depth": (("location", "depth"), depth),
    },
)

ds["aragonite"].attrs["units"] = "mmol/m^3"
ds["aragonite"].attrs["long_name"] = "Aragonite Saturation"
ds["depth"].attrs["units"] = "meters"
ds["depth"].attrs["long_name"] = "Depth Below Sea Surface"
ds["lon"].attrs["units"] = "degrees_east"
ds["lat"].attrs["units"] = "degrees_north"

# Save to NetCDF file
ds.to_netcdf(output_netcdf_path)
logger.info(f"Aragonite field saved to {output_netcdf_path}")