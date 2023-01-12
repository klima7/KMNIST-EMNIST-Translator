import os
import shutil
import tempfile
import os.path
from pathlib import Path

import numpy as np
import streamlit
import cv2


@streamlit.cache
def zip_images(images):
    temp_dir_path = tempfile.mkdtemp()
    images_dir_path = os.path.join(temp_dir_path, 'images')
    os.mkdir(images_dir_path)

    for i, image in enumerate(images):
        path = os.path.join(images_dir_path, f'{i}.png')
        cv2.imwrite(path, image)

    zip_path = os.path.join(temp_dir_path, 'images')
    shutil.make_archive(zip_path, 'zip', images_dir_path)
    return zip_path + '.zip'


def get_uploaded_images(uploaded_files, with_names=False):
    images = []
    uploaded_files = sorted(uploaded_files, key=lambda uf: int(uf.name[:-4]))
    temp_dir_path = Path(tempfile.mkdtemp())

    for file in uploaded_files:
        path = str(temp_dir_path / file.name)

        with open(path, 'wb') as f:
            f.write(file.read())

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        images.append(image)

    images = np.array(images)
    names = [file.name[:-4] for file in uploaded_files]
    
    return (images, names) if with_names else images
