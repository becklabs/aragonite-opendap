import pandas as pd
import joblib
import PyCO2SYS


class TCNModule: ...


class TAlkRegressionModule:
    def __init__(self, checkpoint_path: str):
        self.model = joblib.load(checkpoint_path)

    def predict(self, data: pd.DataFrame, salinity_col: str) -> pd.Series:
        X = data[salinity_col].values.reshape(-1, 1)
        y_pred = self.model.predict(X)
        return pd.Series(y_pred)


class DICRegressionModule:
    SEASONS = {
        "October - March": lambda column: column.dt.month.isin([10, 11, 12, 1, 2, 3]),
        "April - June": lambda column: column.dt.month.isin([4, 5, 6]),
        "July - September": lambda column: column.dt.month.isin([7, 8, 9]),
    }

    def __init__(self, checkpoint_path: str):
        self.checkpoint = joblib.load(checkpoint_path)

    def predict(
        self,
        data: pd.DataFrame,
        salinity_col: str,
        temperature_col: str,
        chlorophyll_col: str,
        time_col: str,
    ):
        result = pd.Series(index=data.index, dtype=float)
        for season in self.SEASONS:
            mask = self.SEASONS[season](data[time_col])
            X = data[mask][[temperature_col, salinity_col, chlorophyll_col]].values
            X = self._transform(X, season=season)
            print(X)
            y_pred, _ = self.checkpoint[season]["model"].predict(X)
            y_pred = self._inverse_transform(y_pred, season=season).reshape(-1)
            result.loc[mask] = y_pred
        return result

    def _transform(self, X, season: str):
        season_checkpoint = self.checkpoint[season]
        Xm = season_checkpoint["Xm"]
        Xs = season_checkpoint["Xs"]
        return (X - Xm) / Xs

    def _inverse_transform(self, y, season: str):
        season_checkpoint = self.checkpoint[season]
        ym = season_checkpoint["ym"]
        ys = season_checkpoint["ys"]
        return y * ys + ym


class CO2SYSAragoniteModule:
    def __init__(self):
        pass

    def predict(
        self,
        data: pd.DataFrame,
        salinity_col: str,
        talk_col: str,
        dic_col: str,
        temperature_col: str,
        depth_col: str,
    ) -> pd.Series:
        co2sys = PyCO2SYS.sys(
            par1_type=1,  # Total Alkalinity input 1
            par1=data[talk_col],
            par2_type=2,  # DIC input 2
            par2=data[dic_col],
            salinity=data[salinity_col],  # practical salinity (default 35)
            temperature=25,  # temperature at which par1 and par2 arguments are provided in °C (default 25 °C)
            pressure=0,  # water pressure at which par1 and par2 arguments are provided in dbar (default 0 dbar)
            temperature_out=data[
                temperature_col
            ],  # temperature at which results will be calculated in °C
            pressure_out=data[
                depth_col
            ],  # water pressure at which results will be calculated in dbar
            opt_pH_scale=4,  # NBS 4
            opt_k_carbonic=1,  # Roy et al 1993 1
            opt_k_bisulfate=1,  # Dickson et al 1990 1
            opt_total_borate=2,  # Lee et al 2010 2
            opt_k_fluoride=1,  # Dickson and Riley 1979 1
        )
        return pd.Series(co2sys["saturation_aragonite_out"])
