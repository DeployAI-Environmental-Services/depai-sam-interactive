import os

WMS_URL = "https://tiles.maps.eox.at/wms"
LAYER = "s2cloudless-2020"
IMAGE_FORMAT = "image/jpeg"
WORK_DIR = os.getenv("SHARED_FOLDER_PATH")
RESOLUTION = 10
TORCH_DEVICE = "cpu"
WEIGHTS_PATH = "./src/weights"
DEBUG = True
