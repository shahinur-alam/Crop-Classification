# scripts/make_datasets.py
import os
import shutil
from sklearn.model_selection import train_test_split
from glob import glob

def split_dataset(raw_dir, output_dir, val_frac=0.1, test_frac=0.1, seed=42):
    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    os.makedirs(output_dir, exist_ok=True)
    for cls in classes:
        # Make class dirs
        os.makedirs(os.path.join(output_dir, 'train', cls), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'val', cls), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test', cls), exist_ok=True)
        # List images
        imgs = glob(os.path.join(raw_dir, cls, '*.jpg'))
        # Split
        train, test = train_test_split(imgs, test_size=test_frac, random_state=seed)
        train, val = train_test_split(train, test_size=val_frac/(1-test_frac), random_state=seed)
        # Copy files
        for f in train:
            shutil.copy(f, os.path.join(output_dir, 'train', cls, os.path.basename(f)))
        for f in val:
            shutil.copy(f, os.path.join(output_dir, 'val', cls, os.path.basename(f)))
        for f in test:
            shutil.copy(f, os.path.join(output_dir, 'test', cls, os.path.basename(f)))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    split_dataset(args.data_dir, args.output_dir)
