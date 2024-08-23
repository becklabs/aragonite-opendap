import numpy as np
import pandas as pd
from pydap.client import open_url


def argclosest(arr, val):
    return np.argmin(np.abs(arr - val))


class AQUAMODISCHLORL3:
    """
    Level 3 Chlorophyll
    https://oceandata.sci.gsfc.nasa.gov/opendap/MODISA/L3SMI/contents.html
    """
    def __init__(self):
        dataset = open_url(self._dataset_url(pd.Timestamp("2024-01-06")))
        self.lats = dataset["lat"][:].data
        self.lons = dataset["lon"][:].data

    def _dataset_url(self, date: pd.Timestamp):
        day = date.strftime("%d")
        month = date.strftime("%m")
        year = date.strftime("%Y")
        return f"dap4://oceandata.sci.gsfc.nasa.gov/opendap/MODISA/L3SMI/{year}/{month + day}/AQUA_MODIS.{year + month + day}.L3m.DAY.CHL.chlor_a.4km.NRT.nc"

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
            url = self._dataset_url(day)
            dataset = open_url(url)

            chlor_a = dataset["chlor_a"][lat_low:lat_high, lon_low:lon_high]
            chlor_a_data = chlor_a.data.squeeze()

            # Check if close to fill value instead
            nans = np.isclose(chlor_a_data, chlor_a["_FillValue"])

            chlor_a_data[nans] = np.nan

            output[i] = chlor_a

        output = output.reshape((n_days, n_lats * n_lons))

        xy = np.meshgrid(self.lons[lon_low:lon_high], self.lats[lat_low:lat_high])
        xy = np.stack(xy, axis=-1).reshape(-1, 2)

        time = pd.date_range(start, end)

        return output, xy, time
