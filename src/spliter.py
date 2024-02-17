from pathlib import Path
from shapely.geometry import Polygon
from loguru import logger
import cv2 as cv
from .make_patch import generate_patches_dict


# 带yolo 数据分割标注的patch，https://docs.ultralytics.com/zh/datasets/segment/
class YOLOImagePatch:
    def __init__(self, patch_dict: dict, name: str,
                 outfolder: Path, annotation_folder: Path):
        if not outfolder.exists():
            outfolder.mkdir(parents=True, exist_ok=True)
        if not annotation_folder.exists():
            annotation_folder.mkdir(parents=True, exist_ok=True)
        self.ymin, self.xmin, self.ymax, self.xmax = patch_dict['loc']
        self.image = patch_dict['image']
        self.annotations = []

        self.img_out_path = outfolder / f"{name}.jpg"
        self.annotation_path = annotation_folder / f'{name}.txt'

    def add_label(self, label: str):
        label_id, *pos = label.split(' ')
        points = [(float(x), float(y)) for x, y in zip(pos[0::2], pos[1::2])]
        inside = True
        new_points = [label_id]
        # 是否都在内部
        for point in points:
            x, y = point
            if y < self.ymin or x < self.xmin or y > self.ymax or x > self.xmax:
                inside = False
                break
        # 如果都在内部
        if inside:
            valid_points = points
        # 如果不是都在内部
        else:
            if len(points) == 3:
                valid_points = []
            else:
                label_polygon = Polygon(points)
                bundary = Polygon(
                    ((self.xmin, self.ymin), (self.xmax, self.ymin), (self.xmax, self.ymax), (self.xmin, self.ymax)))
                try:
                    new_polygon = label_polygon.intersection(bundary)
                except Exception as why:
                    logger.error(f'deyi_info: geom invalid, {why=}, {label}')
                    return {'error': "多边形扭曲"}
                if isinstance(new_polygon, Polygon):
                    valid_points = list(new_polygon.exterior.coords)
                else:
                    valid_points = []
        if valid_points:
            for point in valid_points:
                x, y = point
                ly = y - self.ymin
                lx = x - self.xmin
                new_points += [lx, ly]
            points_line = ' '.join(new_points)
            self.annotations.append(f'{points_line}\n')
        else:
            return {'error': "多边形超出标注框"}

    def save(self):
        # 保存图片和标注文件
        cv.imwrite(str(self.img_out_path), self.image)
        with open(self.annotation_path, 'w') as f:
            for line in self.annotations:
                f.write(line)


# 带yolo 数据分割标注的图像, https://docs.ultralytics.com/zh/datasets/segment/
class YOLOImageSpliter:
    def __init__(self, image_path: Path, patch_size: int = 640, overlap: int = 140):
        self.patch_size = patch_size
        self.overlap = overlap
        self.image_path = image_path
        self.label_path = None

    def split(self):
        """
        利用滑窗分开图像，并分开label
        """
        ...

    def save(self, output_folder: Path) -> None:
        """保存分开的子图和标注文件"""
        raise ValueError('no label for this image')




