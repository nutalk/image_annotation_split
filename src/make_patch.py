import cv2 as cv
from loguru import logger


def tell_diff(local: int, target: int, overlap: int):
    if local <= 640:
        return 640 - local
    else:
        res_target = target - overlap
    diff = res_target - (local - 640) % res_target
    if diff == target:
        return 0
    else:
        return diff


def make_boarder(image, target: int, overlap: int):
    # 填充图片长宽
    h, w, *c = image.shape
    bottom = tell_diff(h, target, overlap)
    right = tell_diff(w, target, overlap)
    logger.info(f'{bottom=}, {right=}')
    image = cv.copyMakeBorder(image, top=0, bottom=bottom,
                              left=0, right=right,
                              borderType=cv.BORDER_CONSTANT, value=1)
    return image


def generate_patches_dict(infile: str, chopsize: int = 640, overlap: int = 140):
    img = cv.imread(infile)
    h, w, *c = img.shape
    logger.info(f'origin shape {h=}, {w=}')
    img = make_boarder(img, chopsize, overlap)
    h, w, *c = img.shape
    logger.info(f'after boarder shape {h=}, {w=}')

    step_size = chopsize - overlap

    output = []
    for y in range(0, h-1, step_size):
        for x in range(0, w-1, step_size):
            yb = y + chopsize
            xr = x + chopsize
            if yb > h or xr > w:
                continue
            patch = img[y:yb, x:xr]
            output.append({'loc': (y, x, yb, xr),
                           'image': patch})
    return output

