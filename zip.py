import argparse
import glob
import multiprocessing as mp
import os
import pathlib
import zipfile
from concurrent.futures import Future, ProcessPoolExecutor, as_completed

import tqdm


def create_zip(file_name: str,
               base_directory: str | pathlib.Path,
               ) -> None:

    base_directory = pathlib.Path(base_directory)

    export_path = file_name+".zip"
    with zipfile.ZipFile(export_path,
                         mode="w",
                         compression=zipfile.ZIP_LZMA,  # zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for file_path in tqdm.tqdm(base_directory.rglob("*")):
            archive.write(
                file_path,
                arcname=file_path
            )


create_zip("dataset", "dataset")
