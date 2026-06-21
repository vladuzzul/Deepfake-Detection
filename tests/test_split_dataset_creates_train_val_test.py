"""
This file confirms if every working image had been split.
"""
import unittest
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DOWNLOADED_IMAGES_PATH = DATA_DIR / "downloaded_images.csv"
TRAIN_PATH = DATA_DIR / "train_df.csv"
VAL_PATH = DATA_DIR / "val_df.csv"
TEST_PATH = DATA_DIR / "test_df.csv"

class TestSplit(unittest.TestCase):
    def test_split(self):
        all_df = pd.read_csv(DOWNLOADED_IMAGES_PATH)
        test_df = pd.read_csv(TEST_PATH)
        val_df = pd.read_csv(VAL_PATH)
        train_df = pd.read_csv(TRAIN_PATH)
        
        split_rows = len(test_df) + len(val_df) + len(train_df)
        all_rows = len(all_df)

        self.assertEqual(split_rows, all_rows)

if __name__ == "__main__":
    unittest.main()