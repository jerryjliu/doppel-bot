"""List files."""

from modal import Image, Stub

from .common import (
    # user_data_path, 
    stub, output_vol, VOL_MOUNT_PATH
)
from .common_sql import user_data_path
import os


@stub.function(
    network_file_systems={VOL_MOUNT_PATH.as_posix(): output_vol},
    cloud="gcp"
)
def main():
    import json

    # read data json
    # tmp = json.load(open(VOL_MOUNT_PATH / "data" / "Jerry Liu" / "data.json"))
    # print(tmp)
    # print(len(tmp))
    # raise Exception

    # # read sql json
    # fp = open(user_data_path("Jerry Liu"), 'r')
    # data = [json.loads(line) for line in fp]
    # print(data[0:2])
    # print(len(data))
    # raise Exception

    print(os.listdir(VOL_MOUNT_PATH / "data_sql" / "Jerry Liu"))