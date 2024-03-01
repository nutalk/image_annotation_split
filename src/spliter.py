from pathlib import Path
from shapely.geometry import Polygon
from loguru import logger
import cv2 as cv
import numpy as np
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
        self.labeled_image = np.copy(self.image)
        self.h, self.w, *c = self.image.shape
        self.annotations = []
        self.draw_annotations = []
        self.img_out_path = outfolder / f"{name}.jpg"
        self.draw_img_path = outfolder / f"{name}_draw.jpg"

        self.annotation_path = annotation_folder / f'{name}.txt'

    def add_label(self, label: str, h: int, w: int):
        label = label.strip()
        label_id, *pos = label.split(' ')
        points = [(float(x) * w, float(y) * h) for x, y in zip(pos[0::2], pos[1::2])]
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
                    logger.error(f'geom invalid, {why=}, {label}')
                    return {'error': "多边形扭曲"}
                if isinstance(new_polygon, Polygon):
                    valid_points = list(new_polygon.exterior.coords)
                else:
                    valid_points = []
        if valid_points:
            draw_annotations = []
            for point in valid_points:
                x, y = point # x,y是全局大图坐标
                ly = min(y - self.ymin, self.h - 1)
                lx = min(x - self.xmin, self.w - 1)
                new_points += [round(lx / self.w, 6), round(ly / self.h, 6)]
                draw_annotations.append([round(lx), round(ly)])
            points_line = ' '.join([str(i) for i in new_points])
            self.annotations.append(f'{points_line}\n')
            self.draw_annotations.append(draw_annotations)
        else:
            return {'error': "多边形超出标注框"}

    def save(self):
        # 保存图片和标注文件
        cv.imwrite(str(self.img_out_path), self.image)
        with open(self.annotation_path, 'w') as f:
            for line in self.annotations:
                f.write(line)
        draw_image = np.copy(self.image)
        for draw in self.draw_annotations:
            logger.info(draw)
            pts = np.array(draw, np.int32)
            # pts = pts.reshape((-1, 1, 2))
            logger.info(pts)
            cv.polylines(draw_image, [pts], True, (0, 255, 0), 2)
        cv.imwrite(str(self.draw_img_path), draw_image)


# 带yolo 数据分割标注的图像, https://docs.ultralytics.com/zh/datasets/segment/
class YOLOImageSpliter:
    def __init__(self, image_path: Path, patch_size: int = 640, overlap: int = 140):
        self.patch_size = patch_size
        self.overlap = overlap
        self.image_path = image_path
        self.image = cv.imread(str(image_path))
        self.name = self.image_path.stem
        train_name = image_path.parent.stem
        self.label_path = image_path.parent.parent.parent / 'labels' / train_name / f"{image_path.stem}.txt"
        self.patches = []
        self.output_img_folder = image_path.parent.parent.parent.parent / 'split' / 'images' / train_name
        self.output_label_folder = image_path.parent.parent.parent.parent / 'split' / 'labels' / train_name

    def split(self):
        """
        利用滑窗分开图像，并分开label
        """
        h, w, *c = self.image.shape
        patch_dict, self.merge_shape = generate_patches_dict(str(self.image_path), self.patch_size, self.overlap)
        for i, patch in enumerate(patch_dict):
            patch_obj = YOLOImagePatch(patch, f'{self.name}_{i}', self.output_img_folder, self.output_label_folder)
            self.patches.append(patch_obj)
        with self.label_path.open('r') as f:
            lines = f.readlines()
            for line in lines:
                for patch in self.patches:
                    patch.add_label(line, h, w)

    def save(self) -> None:
        """保存分开的子图和标注文件"""
        for patch in self.patches:
            patch.save()





