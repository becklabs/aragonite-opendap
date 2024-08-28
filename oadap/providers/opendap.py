import os
from abc import ABC, abstractmethod
import pandas as pd  # type: ignore
import numpy as np
from pydap.client import open_url
from typing import Optional, List, Union
from pydap.cas.urs import setup_session


def argclosest(arr, val):
    return np.argmin(np.abs(arr - val))


def K_to_C(K):
    return K - 273.15


class NASA_OPeNDAP(ABC):
    def __init__(
        self,
        variable: str,
        lat_variable: str = "lat",
        lon_variable: str = "lon",
        time_variable: Optional[str] = None,
        grid_lats: Optional[np.ndarray] = None,
        grid_lons: Optional[np.ndarray] = None,
        scale_factor_attr: Optional[str] = None,
        add_offset_attr: Optional[str] = None,
        fill_value_attr: str = "_FillValue",
        requires_auth: bool = False,
        earthdata_username: Optional[str] = None,
        earthdata_password: Optional[str] = None,
    ):

        self.variable = variable
        self.lat_variable = lat_variable
        self.lon_variable = lon_variable
        self.time_variable = time_variable
        self.scale_factor_attr = scale_factor_attr
        self.add_offset_attr = add_offset_attr
        self.fill_value_attr = fill_value_attr

        self.requires_auth = requires_auth
        self.earthdata_username = earthdata_username
        self.earthdata_password = earthdata_password

        if self.requires_auth:
            if earthdata_username is None or earthdata_password is None:
                try:
                    earthdata_username = os.environ["EARTHDATA_USERNAME"]
                    earthdata_password = os.environ["EARTHDATA_PASSWORD"]
                except KeyError:
                    raise ValueError(
                        "Please set the EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables"
                    )
            self.session = setup_session(earthdata_username, earthdata_password)
        else:
            self.session = None

        self.grid_lats = grid_lats
        self.grid_lons = grid_lons

    @abstractmethod
    def get_granule_url(self, date: pd.Timestamp): ...

    def subset(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ):
        """
        Subset the dataset to a bounding box and time range

        Returns:
        - data (n_days, n_lats * n_lons)
        - xy coordinates (n_lats * n_lons, 2)
        - time range (n_days)
        """

        url = self.get_granule_url(start)
        dataset = open_url(url, session=self.session)
        if self.grid_lats is None or self.grid_lons is None:
            self.grid_lats = dataset[self.lat_variable][:].data
            self.grid_lons = dataset[self.lon_variable][:].data

        lat_low = argclosest(self.grid_lats, lat_min)
        lat_high = argclosest(self.grid_lats, lat_max)
        lat_low, lat_high = min(lat_low, lat_high), max(lat_low, lat_high)

        lon_low = argclosest(self.grid_lons, lon_min)
        lon_high = argclosest(self.grid_lons, lon_max)
        lon_low, lon_high = min(lon_low, lon_high), max(lon_low, lon_high)

        n_days = (end - start).days + 1
        n_lats = lat_high - lat_low
        n_lons = lon_high - lon_low

        output = np.empty((n_days, n_lats, n_lons))

        for i, day in enumerate(pd.date_range(start, end)):
            url = self.get_granule_url(day)
            dataset = open_url(url, session=self.session)

            if self.time_variable is not None:
                variable = dataset[self.variable][0, lat_low:lat_high, lon_low:lon_high]
            else:
                variable = dataset[self.variable][lat_low:lat_high, lon_low:lon_high]

            variable_data = variable.data.squeeze()

            # Replace fill values with NaN
            nans = np.isclose(variable_data, variable.attributes[self.fill_value_attr])
            variable_data = variable_data.astype(np.float32)
            variable_data[nans] = np.nan

            # Apply scale factor and add offset
            if self.scale_factor_attr is not None:
                variable_data *= variable.attributes[self.scale_factor_attr]
                if self.add_offset_attr is not None:
                    variable_data += variable.attributes[self.add_offset_attr]

            output[i] = variable_data

        output = output.transpose((0, 2, 1))

        assert self.grid_lats is not None
        assert self.grid_lons is not None

        xy = np.meshgrid(self.grid_lons[lon_low:lon_high], self.grid_lats[lat_low:lat_high])
        xy = np.stack(xy, axis=-1).transpose(1, 0, 2)

        time = pd.date_range(start, end)

        return output, xy, time


class AQUAMODISCHLORL3(NASA_OPeNDAP):
    """
    Level 3 Chlorophyll
    https://oceandata.sci.gsfc.nasa.gov/opendap/MODISA/L3SMI/contents.html
    """
    def __init__(self):
        super().__init__(variable="chlor_a")

    def get_granule_url(self, date: pd.Timestamp):
        day = date.strftime("%d")
        month = date.strftime("%m")
        year = date.strftime("%Y")
        return f"dap4://oceandata.sci.gsfc.nasa.gov/opendap/MODISA/L3SMI/{year}/{month + day}/AQUA_MODIS.{year + month + day}.L3m.DAY.CHL.chlor_a.4km.NRT.nc"


class GHRSSTL4(NASA_OPeNDAP):
    """
    Level 4 Sea Surface Temperature
    https://cmr.earthdata.nasa.gov/virtual-directory/collections/C1996881146-POCLOUD/temporal
    """

    def __init__(self, earthdata_username: Optional[str] = None, earthdata_password: Optional[str] = None):
        super().__init__(
            variable="analysed_sst",
            time_variable="time",
            scale_factor_attr="scale_factor",
            add_offset_attr="add_offset",
            earthdata_username=earthdata_username,
            earthdata_password=earthdata_password,
            requires_auth=True,
        )

    def get_granule_url(self, date: pd.Timestamp):
        granule = (date + pd.Timedelta(days=1)).strftime("%Y%m%d")
        return f"dap4://opendap.earthdata.nasa.gov/providers/POCLOUD/collections/GHRSST%20Level%204%20MUR%20Global%20Foundation%20Sea%20Surface%20Temperature%20Analysis%20(v4.1)/granules/{granule}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1"
