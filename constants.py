import os

PROJECT_ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

IMAGE_SIZE = (28, 28)
IMAGE_SHIFT = (10, 10)  # add blank space outside images
WORLD_CENTER = [IMAGE_SHIFT[i] + IMAGE_SIZE[i] // 2 for i in range(2)]
