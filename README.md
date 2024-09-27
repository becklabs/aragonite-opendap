# oadap
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

## Overview

This repository implements the framework described in *Data-Driven Modeling of 4D Ocean and Coastal Acidification from Surface Measurements* to predict Aragonite Saturation State ($\Omega_{\text{Ar}}$) fields for Massachusetts Bay. Satellite data sources are accessed dynamically via [PODAAC](https://podaac.jpl.nasa.gov/), enabling up-to-date, daily predictions. The predictions are served over an [OPeNDAP](https://www.earthdata.nasa.gov/engage/open-data-services-and-software/api/opendap) protocol, allowing for easy integration with existing tools and services.

<p align="center">
	<img src="assets/OA_framework.png" alt="photo not available" width="80%" height="80%">
</p>

## Setup


