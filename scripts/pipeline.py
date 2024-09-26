import pandas as pd
import numpy as np
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

# CONSTANTS
WINDOW_SIZE = 20

# VARIABLES
start = pd.Timestamp("2018-01-01")
end = pd.Timestamp("2018-01-30")

data_dir = "data/"
salinity_data_path = data_dir + "MWRA/MWRA_clean.csv"


satellite_xy = np.load(data_dir + "EarthData/sst_grid/xy.npy").reshape(-1, 2)
neighbor_inds = np.load(data_dir + "EarthData/sst_grid/neighbors.npy").reshape(-1, 4)

lat_min = np.min(satellite_xy[:, 1])
lat_max = np.max(satellite_xy[:, 1])
lon_min = np.min(satellite_xy[:, 0])
lon_max = np.max(satellite_xy[:, 0])

satellite_xy = satellite_xy[np.all(neighbor_inds != -1, axis=1)]

chlor_ocean_mask = np.load(data_dir + "EarthData/chlor_grid/ocean_mask.npy")

sal_tcn_config = "config/tcn/v0.yaml"
temp_tcn_config = "config/tcn/v0.yaml"

sal_tcn_checkpoint = "checkpoints/TCN/temperature/model_epoch_95.pth"
temp_tcn_checkpoint = "checkpoints/TCN/salinity/model_epoch_95.pth"

temp_fvcom_artifacts_dir = data_dir + "FVCOM/preprocessed/temperature/all/artifacts/"
sal_fvcom_artifacts_dir = data_dir + "FVCOM/preprocessed/salinity/all/artifacts/"

dic_regression_checkpoint = "checkpoints/DIC_regression/model.pkl"
ta_regression_checkpoint = "checkpoints/TA_regression/model.pkl"

fvcom_siglay = load_mat(data_dir + "FVCOM/siglay.mat")

####

data_start = start - pd.Timedelta(days=WINDOW_SIZE - 1)
data_end = end

# FETCHING DATA
salinity_provider = MWRASalinity(path=salinity_data_path)
temperature_provider = GHRSSTL4()
chlorophyll_provider = AQUAMODISCHLORL3()

sal_raw, sal_xy, sal_time = salinity_provider.subset(
    lat_min=lat_min,
    lat_max=lat_max,
    lon_min=lon_min,
    lon_max=lon_max,
    start=pd.Timestamp("2015-01-01"),  # Fetch all salinity data
    end=pd.Timestamp("2025-01-01"),
)

sst_raw, sst_xy, sst_time = temperature_provider.subset(
    lat_min=lat_min,
    lat_max=lat_max,
    lon_min=lon_min,
    lon_max=lon_max,
    start=data_start,
    end=data_end,
)

chlor_raw, chlor_xy, chlor_time = chlorophyll_provider.subset(
    lat_min=lat_min,
    lat_max=lat_max,
    lon_min=lon_min,
    lon_max=lon_max,
    start=data_start - pd.Timedelta(days=1),
    end=data_end + pd.Timedelta(days=1),
)

# INTERPOLATION
chlor_interpolator = TimeSubsetInterpolator(
    n_days=1,
    start_date=data_start,
    end_date=data_end,
    interpolator_factory=lambda: KrigingInterpolator(variogram_model="exponential"),
)
chlor_raw_swath, chlor_xy_swath, chlor_time_swath = grid_to_swath(
    grid=chlor_xy[chlor_ocean_mask], values=chlor_raw[chlor_ocean_mask], time=chlor_time
)

chlor_interpolator.fit(
    xy=np.log(chlor_xy_swath), values=chlor_raw_swath, time=chlor_time_swath
)

chlor_surface = np.exp(
    chlor_interpolator.predict(
        grid=satellite_xy, days=pd.date_range(data_start, data_end)
    )
)

sal_interpolator = ScaledKrigingInterpolator(variogram_model="linear")

sal_interpolator.fit(xy=sal_xy, values=sal_raw, time=sal_time)

salinity_surface = sal_interpolator.predict(
    grid=satellite_xy, days=pd.date_range(data_start, data_end)
)


# RECONSTRUCT TEMP FIELD
fvcom_xy = np.load(temp_fvcom_artifacts_dir + "xy.npy")
h = np.load(temp_fvcom_artifacts_dir + "h.npy")
phi = np.load(temp_fvcom_artifacts_dir + "phi.npy")
climatologies = np.load(temp_fvcom_artifacts_dir + "climatology_days.npy")

temperature_tcn = TCNModule(
    grid_xy=satellite_xy,
    grid_neighbor_inds=neighbor_inds,
    fvcom_xy=fvcom_xy,
    fvcom_h=h,
    fvcom_phi=phi,
    fvcom_climatologies=climatologies,
    config_path=temp_tcn_config,
    checkpoint_path=temp_tcn_checkpoint,
)

temp_field, output_grid, _ = temperature_tcn.predict(
    surface=sst_raw.reshape(-1, sst_raw.shape[-1]),
    time=sst_time,
)

# RECONSTRUCT SALINITY FIELD
fvcom_xy = np.load(sal_fvcom_artifacts_dir + "xy.npy")
h = np.load(sal_fvcom_artifacts_dir + "h.npy")
phi = np.load(sal_fvcom_artifacts_dir + "phi.npy")
climatologies = np.load(sal_fvcom_artifacts_dir + "climatology_days.npy")

salinity_tcn = TCNModule(
    grid_xy=satellite_xy,
    grid_neighbor_inds=neighbor_inds,
    fvcom_xy=fvcom_xy,
    fvcom_h=h,
    fvcom_phi=phi,
    fvcom_climatologies=climatologies,
    config_path=sal_tcn_config,
    checkpoint_path=sal_tcn_checkpoint,
)

salinity_field, _, _ = salinity_tcn.predict(
    surface=salinity_surface,
    time=sst_time,
)

# DIC REGRESSION
dic_regressor = DICRegressionModule(checkpoint_path=dic_regression_checkpoint)

dic_field = dic_regressor.predict(
    salinity_field=salinity_field,
    temperature_field=temp_field,
    chlorophyll_surface=chlor_surface,
    time=pd.date_range(data_start, data_end),
)

# TAlk REGRESSION
talk_regressor = TAlkRegressionModule(checkpoint_path=ta_regression_checkpoint)

talk_field = talk_regressor.predict(
    salinity=salinity_field,
)

# CO2SYS ARAGONITE
co2sys_module = CO2SYSAragoniteModule()

aragonite_field = co2sys_module.predict(
    salinity_field=salinity_field,
    temperature_field=temp_field,
    dic_field=dic_field,
    talk_field=talk_field,
    depth=h * -fvcom_siglay,
)
