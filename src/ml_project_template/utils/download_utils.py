"""Utility functions aimed at downloading any data from external sources."""

import cloudpathlib

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def download_kaggle_dataset(dataset_name: str, output_folder: str) -> None:
    """Download a given Kaggle dataset.

    Args:
        dataset_name: for example googleai/pfam-seed-random-split
        output_folder: where the data downloaded will be stored (ideally data/ folder)
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    log.info("Authenticating to Kaggle API")
    api.authenticate()
    log.info("Downloading dataset")
    api.dataset_download_files(dataset_name, path=output_folder, unzip=True, quiet=False)
    log.info("Download successful")


def download_cloud_directory(cloud_directory: str, output_folder: str, cloud: str = "gs") -> None:
    """Download a given cloud directory.

    Args:
        cloud_directory: for example gs://bucket-name/path/to/directory
        output_folder: where the data downloaded will be stored (ideally data/ folder)
        cloud: the cloud provider, currently only "gs" is supported
    """
    cloudpathlib.Path(cloud_directory).download_to(output_folder)
