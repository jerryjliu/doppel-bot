"""List files."""

from modal import Image, Stub

from .common import user_data_path, stub, output_vol, VOL_MOUNT_PATH
import os


@stub.function(
    network_file_systems={VOL_MOUNT_PATH.as_posix(): output_vol},
    cloud="gcp"
)
def main():
    import json
    tmp = json.load(open(VOL_MOUNT_PATH / "data" / "Jerry Liu" / "data.json"))
    print(tmp)
    print(len(tmp))
    raise Exception