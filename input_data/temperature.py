import os
import numpy as np
import pandas as pd
from pydap.cas.urs import setup_session
from pydap.client import open_url
from typing import Optional


def argclosest(arr, val):
    return np.argmin(np.abs(arr - val))


def K_to_C(K):
    return K - 273.15


class GHRSSTL4:
    """
    Level 4 Sea Surface Temperature
    https://cmr.earthdata.nasa.gov/virtual-directory/collections/C1996881146-POCLOUD/temporal
    """

    def __init__(
        self,
        earthdata_username: Optional[str] = None,
        earthdata_password: Optional[str] = None,
    ):
        if earthdata_username is None or earthdata_password is None:
            try:
                earthdata_username = os.environ["EARTHDATA_USERNAME"]
                earthdata_password = os.environ["EARTHDATA_PASSWORD"]
            except KeyError:
                raise ValueError(
                    "Please set the EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables"
                )

        self.dataset_url = "dap4://opendap.earthdata.nasa.gov/providers/POCLOUD/collections/GHRSST%20Level%204%20MUR%20Global%20Foundation%20Sea%20Surface%20Temperature%20Analysis%20(v4.1)/granules/{granule}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1"
        self.session = setup_session(earthdata_username, earthdata_password)

        self.lats = np.arange(-89.99, 90, 0.01)
        self.lons = np.arange(-179.99, 180, 0.01)

    def get(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ):

        lat_low = argclosest(self.lats, lat_min)
        lat_high = argclosest(self.lats, lat_max)

        lon_low = argclosest(self.lons, lon_min)
        lon_high = argclosest(self.lons, lon_max)

        n_days = (end - start).days + 1
        n_lats = lat_high - lat_low
        n_lons = lon_high - lon_low

        output = np.empty((n_days, n_lats, n_lons))

        for i, day in enumerate(pd.date_range(start, end)):
            granule = (day + pd.Timedelta(days=1)).strftime("%Y%m%d")

            url = self.dataset_url.format(granule=granule)
            dataset = open_url(url, session=self.session)

            sst = dataset["analysed_sst"][0, lat_low:lat_high, lon_low:lon_high]
            sst_data = sst.data.squeeze()

            nans = sst_data == sst.attributes["_FillValue"]
            sst_data = sst_data.astype(np.float32)
            sst_data[nans] = np.nan

            sst_data = (
                sst_data * sst.attributes["scale_factor"] + sst.attributes["add_offset"]
            )  # Scale and offset
            sst_data = K_to_C(sst_data)  # Convert to Celsius

            output[i] = sst_data

        output = output.reshape((n_days, n_lats * n_lons))

        xy = np.meshgrid(self.lons[lon_low:lon_high], self.lats[lat_low:lat_high])
        xy = np.stack(xy, axis=-1).reshape(-1, 2)

        time = pd.date_range(start, end)

        return output, xy, time
