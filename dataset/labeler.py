import json
import os

import cv2 as cv
from rich.progress import track

IMAGE_DIRS = "./data/chopped"
TARGET_FILE = "./labels.json"
IMTYPE = "png"


def get_label(char: str):
    print(char)
    char = ord(char)
    if ord("0") <= char <= ord("9"):
        return char - ord("0")
    if ord("A") <= char <= ord("Z"):
        return char + 10 - ord("A")
    if ord("a") <= char <= ord("z"):
        return char + 36 - ord("a")
    raise ValueError("char is not valid")


capslock = False


def process(path: str, is_cap: bool = None):
    global capslock

    if is_cap is None:
        is_cap = capslock

    img = cv.imread(path, 0)
    assert img is not None
    cv.imshow("image", img)
    cat = chr(cv.waitKey(0))

    if ord(cat) == 27:  # esc
        return -2
    if ord(cat) == 225:
        return process(path, not is_cap)

    if ord(cat) == 229:
        capslock = not capslock
        return process(path)

    try:
        if cat == " ":
            return -1
        if is_cap:
            cat = chr(ord(cat) + ord("A") - ord("a"))
        return get_label(cat)
    except ValueError as e:
        print(f"Error: {e}, retrying...")
        process(path)


def save_file(data: dict, target_file: str):
    data = json.dumps(data)
    with open(target_file, "wt") as f:
        f.write(data)


image_paths = list(filter(lambda x: x.endswith(IMTYPE), os.listdir(IMAGE_DIRS)))

labeled = {}
if os.path.exists(TARGET_FILE):
    with open(TARGET_FILE, "rt") as f:
        labeled = json.loads(f.read())

for i in track(image_paths):
    if i in labeled:
        continue
    category = process(os.path.join(IMAGE_DIRS, i))
    if category == -2:
        print("user aborted")
        break
    labeled[i] = category
    save_file(labeled, TARGET_FILE)
