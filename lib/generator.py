from typing import Tuple, List

import numpy as np
from skimage import transform
from tqdm import tqdm
from einops import rearrange

from .data import MAPPINGS


class PageGenerator:
    def __init__(
        self,
        text: str,
        dataset: Tuple[np.ndarray, np.ndarray],
        h: int = 114,
        w: int = 80,
        corruption_p: float = 0.3,
        salt_p: float = 0.025,
        rotation_radius: int = 30,
        max_scale: float = 1.15,
        unique_characters_per_class: int = None
    ) -> None:
        self.h = h
        self.w = w
        self.text = text
        self.corruption_p = corruption_p
        self.dataset = dataset
        self.rotation_radius = rotation_radius
        self.salt_p = salt_p
        self.max_scale = max_scale
        self.unique_characters_per_class = unique_characters_per_class
        self.prepared_dataset = self.__prepare_dataset()

    def generate_pages(self, max_pages: int = None) -> np.ndarray:
        pages: List[np.ndarray] = []
        pages_text: List[np.ndarray] = []
        text = self.text

        with tqdm(total=len(self.text)) as pbar:
            while len(text) > 0:
                old_len = len(text)
                page, text, page_text = self.__generate_page(text)
                pages_text.append(page_text)
                pages.append(page)
                pbar.update(old_len - len(text))

                if max_pages is not None and len(pages) >= max_pages:
                    break

        return np.asarray(pages), np.asarray(pages_text)

    def __image_corruption(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() <= self.corruption_p:
            degree = np.random.randint(-self.rotation_radius, self.rotation_radius + 1)
            image = transform.rotate(image, degree)
        if np.random.random() <= self.corruption_p:
            h, w = image.shape
            nh, nw = 1, 1
            if np.random.choice([True, False]):
                nh = np.random.uniform(1, self.max_scale)
            else:
                nw = np.random.uniform(1, self.max_scale)
            image = transform.rescale(image, (nh, nw))
            hcrop = int((h * nh - h) / 2)
            wcrop = int((w * nw - w) / 2)
            image = image[hcrop: h + hcrop, wcrop: w + wcrop]
        return image > 0.5

    def __prepare_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.unique_characters_per_class is None:
            return self.dataset

        data_images, data_labels = self.dataset
        new_images, new_labels = [], []
        for char_no in set(MAPPINGS.values()):
            if char_no < 0:
                continue
            indexes = np.where(data_labels == char_no)[0]
            few_indexes = np.random.choice(indexes, size=self.unique_characters_per_class, replace=False)
            new_images.append(data_images[few_indexes])
            new_labels.append(data_labels[few_indexes])

        return np.concatenate(new_images), np.concatenate(new_labels)

    def __generate_page(self, txt: str) -> Tuple[np.ndarray, str]:
        page_data: np.ndarray = np.zeros((self.h, self.w, 32, 32))
        salt = np.random.choice(
            [0, 1], size=page_data.shape, p=[1 - self.salt_p, self.salt_p]
        )
        pepper = np.random.choice(
            [1, 0], size=page_data.shape, p=[1 - self.salt_p, self.salt_p]
        )
        data_images, data_labels = self.prepared_dataset
        i = 0
        gen_txt = ''
        while i < self.h:
            j = 0
            while j < self.w:
                if len(txt) == 0:
                    gen_txt += ' ' * (self.w * (self.h - i - 1) + (self.w - j))
                    i = j = np.inf
                    break
                char = txt[0]
                txt = txt[1:]
                label = MAPPINGS[char]
                if char == "\n":
                    i += 1
                    gen_txt += " " * (self.w - j)
                    j = 0
                    if i == self.h:
                        break
                    continue
                if char == " ":
                    j += 1
                    gen_txt += " "
                    continue
                index = np.random.choice(np.where(data_labels == label)[0])
                image = self.__image_corruption(data_images[index])
                page_data[i, j] = image
                gen_txt += char
                j += 1
            i += 1
        
        gen_txt = np.array([ord(c) for c in gen_txt]).reshape((self.h, self.w))
        return np.clip(pepper - page_data + salt, 0, 1), txt, gen_txt


def generate_pages(text, dataset, corruption_prob, salt_prob, rotation, max_scale, unique_characters):
    pages_data, text_pages = PageGenerator(
        text,
        dataset,
        corruption_p=corruption_prob,
        salt_p=salt_prob,
        rotation_radius=rotation,
        max_scale=max_scale,
        unique_characters_per_class=unique_characters
    ).generate_pages()

    chars_pages = [reshape_page_data_to_image(page_data) for page_data in pages_data]
    text_pages = [page for page in text_pages]
    return chars_pages, text_pages


def reshape_page_data_to_image(page_data: np.ndarray, h=32, w=32) -> np.ndarray:
    return rearrange(page_data, 'H W h w -> (H h) (W w)', h=h, w=w)


def reshape_image_to_page_data(image: np.ndarray, h=32, w=32) -> np.ndarray:
    return rearrange(image, '(H h) (W w) -> H W h w', h=h, w=w)
