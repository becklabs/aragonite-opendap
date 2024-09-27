# aragonite-opendap
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

## Overview

This repository implements the framework described in *Data-Driven Modeling of 4D Ocean and Coastal Acidification from Surface Measurements* to predict Aragonite Saturation State ($\Omega_{\text{Ar}}$) fields for Massachusetts Bay. Satellite data sources are accessed dynamically via [PODAAC](https://podaac.jpl.nasa.gov/), enabling up-to-date, daily predictions. The predictions are served over an [OPeNDAP](https://www.earthdata.nasa.gov/engage/open-data-services-and-software/api/opendap) protocol, allowing for easy integration with existing tools and services.

<p align="center">
	<img src="assets/OA_framework.png" alt="photo not available" width="80%" height="80%">
</p>

## Setup 

#### Install from source:
1. Clone the repository and move into its directory:

```bash
git clone https://github.com/becklabs/aragonite-opendap.git && cd aragonite-opendap
```

2. From this directory, install the `oadap` pip package in editable mode:

```
pip install -e .
```

## Inference
To make $\Omega_{\text{Ar}}$ predictions across a given date range, run the following command:

```
python -m scripts.run_framework \
    --start         [Required] [str]  Start date in YYYY-MM-DD format \
    --end           [Required] [str]  End date in YYYY-MM-DD format \
    --cache_dir     [Optional] [str]  Cache directory for joblib memory (default: intermediate_cache/) \
    --output_nc     [Optional] [str]  Output NetCDF file path (default: aragonite_field.nc)
```
