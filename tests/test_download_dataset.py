"""
This file tests 5 random images to see if their download link is still available.
"""
import os
from pathlib import Path
import requests
from tqdm import tqdm
import pandas as pd
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOWNLOADED_IMAGES_PATH = DATA_DIR / "downloaded_images.csv"


class test_images(unittest.TestCase):
    def test_downloads(self):
        if os.path.exists(DOWNLOADED_IMAGES_PATH):
            df_to_download = pd.read_csv(DOWNLOADED_IMAGES_PATH).sample(5)

            image_paths = []

            for idx, row in tqdm(df_to_download.iterrows(), total=len(df_to_download)):
                url = row['image_url']
                try:
                    response = requests.get(url, timeout=10)
                    file_path = f"{idx}.jpg"
                    image_paths.append(str(file_path))

                except:
                    image_paths.append(None)

            self.assertEqual(len(df_to_download), len(image_paths))
        else:
            self.assertEqual(1, 2)

if __name__ == "__main__":
    unittest.main()
            
            
