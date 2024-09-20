"""
it tries to parse raw images, remove noises and split it into charachters
"""

import multiprocessing.pool
import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 
from rich.progress import track 
import os 
import multiprocessing


def bfs(img, start, visited):
    visited[start[0], start[1]] = True
    points = [start]
    h, w = img.shape
    ind = 0
    while ind < len(points):
        x, y = points[ind]
        ind += 1

        for i in range(max(0, x - 1), min(x + 2, h)):
            for j in range(max(0, y - 1), min(y + 2, w)):
                neighbour = (i, j)
                if img[i, j] != 0 and not visited[i, j]:
                    visited[i, j] = True
                    points.append(neighbour)
    return points
def find_components(img):
    h, w = img.shape
    visited = np.full((h, w), False)
    sets = []
    for i in range(h):
        for j in range(w):
            point = (i, j)
            if img[i, j] != 0 and not visited[i, j]:
                seg = bfs(img, point, visited)
                sets.append(seg) 
    return sets 


to_sub = None
count = 100
for i in range(count):
    img = cv.imread(f"./data/raw/{i}.png")
    img_part = img[45:, 40:]
    last_sum = to_sub.sum() if to_sub is not None else 0
    to_sub = img_part if to_sub is None else to_sub + img_part
    to_sub = to_sub.astype("uint64")
    assert last_sum + img_part.sum() == to_sub.sum(), "probably overflow occured"
    

to_sub = to_sub / count
to_sub = to_sub.astype("uint8")

def get_image(path: str, show: bool = False):
    img = cv.imread(path)
    if show:
        plt.figure()
        plt.imshow(img)
        plt.title("original")
    h, w, _ = to_sub.shape
    for row in range(h):
        for col  in range(w):
            if abs((img[row + 45, col + 40] - to_sub[row, col]).mean()) < 50:
                img[row + 45, col + 40] = 255

    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    original_img = img.copy()
    img = cv.medianBlur(img, 3)
    
    if show:
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.title("blur")
    
    img = cv.bitwise_not(img)
    img = cv.addWeighted(img, 1.80, original_img, -0.55, 0)
    _, img = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
    if show:
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.title("binary")


    components = find_components(img)

    for i in components:
        if len(i) < 20:
            for p in i:
                img[p[0], p[1]] = 0
    if show:
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.title("low segment removed")

    return img

def chop(img, filename: str, do_write: bool = False, destination_dir: str = "/tmp", show: bool = False):
    img = img.copy()
    filename, file_extension = filename.split(".")
    if show:
        hist_data = []
        for i in img.T:
            hist_data.append(np.sum(i))
        plt.plot(hist_data)

    points = []
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] == 255:
                points.append((i, j))
    flags = cv.KMEANS_RANDOM_CENTERS
    _, _, centers = cv.kmeans(
        np.array(points, dtype=np.float32),
        5,
        None,
        (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10,
        flags,
    )

    if show:
        for i in centers:
            plt.axvline(x=i[1], color='r')

    for part_index, center in enumerate(centers):
        y, x = center
        Y, X = img.shape
        HALF_SIZE = 14
        if x > X - HALF_SIZE:
            x = X - HALF_SIZE
        if y > Y - HALF_SIZE:
            y = Y - HALF_SIZE
        if x < HALF_SIZE:
            x = HALF_SIZE
        if y < HALF_SIZE:
            y = HALF_SIZE


        if do_write:
            part = img[int(y) - HALF_SIZE: int(y) + HALF_SIZE,
                       int(x) - HALF_SIZE: int(x) + HALF_SIZE]
            cv.imwrite(f'{destination_dir}/{filename}-{part_index}.{file_extension}', part)
        
        if not do_write:
            cv.rectangle(img, (int(x) - 13, int(y) - 13),
                            (int(x) + 13, int(y) + 13), (255, 0, 0), 1)
        
    return img


def chop_with_path(args):
    """
    since I want to run it with multiprocessing.Pool, I had to just get args and expand it myself.
    """
    if len(args) != 2:
        raise ValueError("I need path and destination dir")
    path, destination_dir = args
    filename = path.split("/")[-1]
    chop(get_image(path), filename, do_write=True, destination_dir=destination_dir)

def chop_images(source_dir, imtype, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    if not os.path.isdir(destination_dir):
        raise ValueError("destination directiry exists but it is not a directory")
    
    images = list(filter(lambda x: x.endswith(f".{imtype}"), os.listdir(source_dir)))
    image_paths = [os.path.join(source_dir, i) for i in images]
    args = [(i, destination_dir) for i in image_paths]


    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for _ in track(pool.imap_unordered(chop_with_path, args), total = len(args)):
        pass

if __name__ == "__main__":
    chop_images("./data/raw", "png", "./data/chopped")