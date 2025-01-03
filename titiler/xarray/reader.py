import contextlib
import os
import pickle
import re
from typing import Any, Dict, List, Optional

import attr
import fsspec
import numpy
import requests
import s3fs
import xarray
from morecantile import TileMatrixSet
from rasterio.crs import CRS
from rio_tiler.constants import WEB_MERCATOR_TMS, WGS84_CRS
from rio_tiler.io.xarray import XarrayReader
from rio_tiler.types import BBox
from starlette.requests import Request
from starlette.exceptions import HTTPException

from titiler.xarray.redis_pool import get_redis
from titiler.xarray.settings import ApiSettings

api_settings = ApiSettings()
cache_client = get_redis()

# ----------------------------------------------------------------
# 1) CONFIG for ephemeral S3 credentials
# ----------------------------------------------------------------

CREDENTIALS_ENDPOINT = os.getenv(
    "CREDENTIALS_ENDPOINT", 
    "https://dev.eodatahub.org.uk/api/workspaces/s3/credentials"
)
DEFAULT_REGION = os.getenv("AWS_REGION", "eu-west-2")

WHITELIST_PATTERNS = [
    r"^https://workspaces-eodhp-[\w-]+\.s3\.eu-west-2\.amazonaws\.com/",
    r"^s3://workspaces-eodhp-[\w-]+/",
]

def force_no_irsa() -> None:
    """
    Remove environment variables that cause botocore/boto3 to attempt
    STS AssumeRoleWithWebIdentity for IRSA (EKS). This ensures we rely
    ONLY on the ephemeral credentials we fetch below.
    """
    for var in ("AWS_ROLE_ARN", "AWS_WEB_IDENTITY_TOKEN_FILE"):
        if var in os.environ:
            del os.environ[var]

def is_whitelisted_url(url: str) -> bool:
    """
    Check if the given URL matches ANY of the patterns in WHITELIST_PATTERNS.
    If yes, we want to attach ephemeral S3 credentials.
    """
    for pattern in WHITELIST_PATTERNS:
        if re.match(pattern, url):
            return True
    return False

def rewrite_https_to_s3_if_needed(url: str) -> str:
    """
    If the URL starts with https://workspaces-eodhp-*, rewrite to s3:// if it matches the pattern.
    Otherwise, return it unchanged.
    """
    https_pattern = r"^https://(workspaces-eodhp-[\w-]+)\.s3\.eu-west-2\.amazonaws\.com/(.*)"
    match = re.match(https_pattern, url)
    if match:
        bucket_part = match.group(1)  # e.g. workspaces-eodhp-dev
        key_part = match.group(2)    # e.g. path/to/file.zarr
        return f"s3://{bucket_part}/{key_part}"
    return url

def fetch_ephemeral_creds(auth_header: str, cookie_header: str) -> Dict[str, str]:
    """
    Call CREDENTIALS_ENDPOINT with user auth headers to get ephemeral AWS creds.
    """
    if not auth_header and not cookie_header:
        raise HTTPException(status_code=401, detail="User not logged in")

    # Force ignoring IRSA if we want ephemeral credentials
    force_no_irsa()

    resp = requests.get(
        CREDENTIALS_ENDPOINT,
        headers={"Authorization": auth_header, "Cookie": cookie_header},
        timeout=5,
    )
    if resp.status_code != 200:
        raise HTTPException(
            status_code=403,
            detail="Unable to fetch ephemeral AWS credentials",
        )
    return resp.json()

# ----------------------------------------------------------------
# 2) CORE UTILS 
# ----------------------------------------------------------------

def parse_protocol(src_path: str, reference: Optional[bool] = False):
    """
    Parse protocol from path.
    """
    match = re.match(r"^(s3|https|http)", src_path)
    protocol = "file"
    if match:
        protocol = match.group(0)
    # override protocol if reference
    if reference:
        protocol = "reference"
    return protocol

def xarray_engine(src_path: str):
    """
    Parse xarray engine from path based on file extension.
    """
    lower_filename = src_path.lower()
    if any(lower_filename.endswith(ext) for ext in [".nc", ".nc4"]):
        return "h5netcdf"
    else:
        return "zarr"


def xarray_open_dataset(
    src_path: str,
    request: Optional[Request] = None,
    reference: bool = False,
    decode_times: bool = True,
    consolidated: bool = True,
    group: Optional[Any] = None,
) -> xarray.Dataset:
    """Open dataset with ephemeral S3 credentials if needed."""

    src_path = rewrite_https_to_s3_if_needed(src_path)

    # Check if in our "workspace" pattern => ephemeral creds
    creds_dict: Dict[str, str] = {}
    if is_whitelisted_url(src_path):
        if request is None:
            raise HTTPException(
                status_code=400,
                detail="No Request object to fetch auth headers from"
            )
        auth_header = request.headers.get("authorization")
        cookie_header = request.headers.get("cookie")
        # Fetch ephemeral
        creds_dict = fetch_ephemeral_creds(auth_header, cookie_header)

    protocol = parse_protocol(src_path, reference=reference)
    xr_engine = xarray_engine(src_path)

    if protocol == "reference":
        reference_args = {"fo": src_path}
        fs = fsspec.filesystem("reference", **reference_args).get_mapper("")
    elif protocol == "s3":
        if creds_dict:
            fs = s3fs.S3FileSystem(
                key=creds_dict["accessKeyId"],
                secret=creds_dict["secretAccessKey"],
                token=creds_dict["sessionToken"],
                client_kwargs={"region_name": DEFAULT_REGION},
            )
            file_handler = (
                fs.open(src_path) if xr_engine == "h5netcdf" else fs.get_mapper(src_path)
            )
        else:
            # no ephemeral => presumably public
            s3_filesystem = s3fs.S3FileSystem()
            file_handler = (
                s3_filesystem.open(src_path)
                if xr_engine == "h5netcdf"
                else s3_filesystem.get_mapper(src_path)
            )
    elif protocol in ["https", "http", "file"]:
        filesystem = fsspec.filesystem(protocol)
        file_handler = (
            filesystem.open(src_path)
            if xr_engine == "h5netcdf"
            else filesystem.get_mapper(src_path)
        )
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")


    xr_open_args: Dict[str, Any] = {
        "decode_coords": "all",
        "decode_times": decode_times,
    }
    if isinstance(group, int):
        xr_open_args["group"] = group

    if xr_engine == "h5netcdf":
        xr_open_args["engine"] = "h5netcdf"
        xr_open_args["lock"] = False
        ds = xarray.open_dataset(file_handler, **xr_open_args)
    else:
        # zarr
        xr_open_args["engine"] = "zarr"
        xr_open_args["consolidated"] = consolidated
        if reference:
            xr_open_args["consolidated"] = False
            xr_open_args["backend_kwargs"] = {"consolidated": False}
        ds = xarray.open_dataset(file_handler, **xr_open_args)

    return ds

# ----------------------------------------------------------------
# 4) Helpers for reading a variable
# ----------------------------------------------------------------

def arrange_coordinates(da: xarray.DataArray) -> xarray.DataArray:
    """
    Arrange coordinates so that we have time/y/x, or y/x.
    """
    if "x" not in da.dims and "y" not in da.dims:
        # Attempt some best-guess renaming for lat/lon
        latitude_var_name = "lat"
        longitude_var_name = "lon"
        for cand_lat in ["lat", "latitude", "LAT", "Latitude"]:
            if cand_lat in da.dims:
                latitude_var_name = cand_lat
                break
        for cand_lon in ["lon", "longitude", "LON", "Longitude"]:
            if cand_lon in da.dims:
                longitude_var_name = cand_lon
                break
        da = da.rename({latitude_var_name: "y", longitude_var_name: "x"})

    # reorder
    if "time" in da.dims:
        da = da.transpose("time", "y", "x")
    else:
        da = da.transpose("y", "x")
    return da

def get_variable(
    ds: xarray.Dataset,
    variable: str,
    datetime: Optional[str] = None,
    drop_dim: Optional[str] = None,
) -> xarray.DataArray:
    """Get Xarray variable as DataArray."""
    da = ds[variable]
    da = arrange_coordinates(da)

    if drop_dim:
        dim_to_drop, dim_val = drop_dim.split("=")
        da = da.sel({dim_to_drop: dim_val}).drop(dim_to_drop)
        da = arrange_coordinates(da)

    # Make sure we have a valid CRS
    crs = da.rio.crs or "epsg:4326"
    da.rio.write_crs(crs, inplace=True)

    # If crossing 180 -> shift
    if crs == "epsg:4326" and (da.x > 180).any():
        da = da.assign_coords(x=(da.x + 180) % 360 - 180)
        da = da.sortby(da.x)

    # If there's a time dimension, pick the first time or nearest
    if "time" in da.dims:
        if datetime:
            time_as_str = datetime.split("T")[0]
            if da["time"].dtype == "O":
                da["time"] = da["time"].astype("datetime64[ns]")
            da = da.sel(
                time=numpy.array(time_as_str, dtype=numpy.datetime64), method="nearest"
            )
        else:
            da = da.isel(time=0)

    return da


@attr.s
class ZarrReader(XarrayReader):
    """ZarrReader: Open Zarr file and access DataArray."""

    src_path: str = attr.ib()
    variable: Optional[str] = attr.ib(default=None)

    # xarray.Dataset options
    request: Optional[Request] = attr.ib(default=None)
    reference: bool = attr.ib(default=False)
    decode_times: bool = attr.ib(default=False)
    group: Optional[Any] = attr.ib(default=None)
    consolidated: Optional[bool] = attr.ib(default=True)

    # xarray.DataArray options
    datetime: Optional[str] = attr.ib(default=None)
    drop_dim: Optional[str] = attr.ib(default=None)

    tms: TileMatrixSet = attr.ib(default=WEB_MERCATOR_TMS)
    geographic_crs: CRS = attr.ib(default=WGS84_CRS)

    ds: xarray.Dataset = attr.ib(init=False)
    input: xarray.DataArray = attr.ib(init=False)

    bounds: BBox = attr.ib(init=False)
    crs: CRS = attr.ib(init=False)

    _minzoom: int = attr.ib(init=False, default=None)
    _maxzoom: int = attr.ib(init=False, default=None)

    _dims: List[str] = attr.ib(init=False, factory=list)
    _ctx_stack = attr.ib(init=False, factory=contextlib.ExitStack)

    def __attrs_post_init__(self):
        """Open dataset, get variable."""
        self.ds = self._ctx_stack.enter_context(
            xarray_open_dataset(
                self.src_path,
                request=self.request,
                reference=self.reference,
                decode_times=self.decode_times,
                consolidated=self.consolidated,
                group=self.group,
            )
        )
        if self.variable:
            self.input = get_variable(
                self.ds,
                self.variable,
                datetime=self.datetime,
                drop_dim=self.drop_dim,
            )
            self.bounds = tuple(self.input.rio.bounds())
            self.crs = self.input.rio.crs
            self._dims = [
                d for d in self.input.dims if d not in [self.input.rio.x_dim, self.input.rio.y_dim]
            ]
        else:
            pass

    def close(self):
        """Close xarray dataset."""
        self.ds.close()

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit."""
        self.close()
