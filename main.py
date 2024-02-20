import argparse
from loguru import logger
import sys
from pathlib import Path

from src.spliter import YOLOImageSpliter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_folder", help="图片所在文件夹")
    parser.add_argument("--logger_level", type=str, default='INFO',
                        choices=['CRITICAL', 'ERROR', 'WARNING', 'SUCCESS', 'INFO', 'DEBUG'])

    args = parser.parse_args()
    run_config_dict = vars(args)

    logger.remove()  # 删除默认的handler
    logger.add(sys.stderr, level=run_config_dict["logger_level"])  # 控制台输出级别

    fd = Path(run_config_dict["image_folder"])
    for img in fd.glob('**/*.png'):
        logger.info(f"Processing {img}")

        spliter = YOLOImageSpliter(img)
        spliter.split()
        spliter.save()


