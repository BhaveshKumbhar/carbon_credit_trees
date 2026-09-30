"""
Scans every image under tree_images/ and verifies it's actually a valid,
openable image — not an HTML error page or other non-image content that
got saved with a .jpg extension (this happened because the downloader
didn't check response content before writing to disk — fixed in v6).

Invalid files are moved to corrupted_flagged/<species>/ rather than
deleted, so you can spot-check what Wikimedia's rejection pages look like
if curious. After running this, run reconcile_metadata.py to sync the
metadata CSV (it already checks whether each file still exists, so moved
files get automatically dropped from there).

Usage:
    python clean_corrupt_images.py
"""

import os
import shutil
from PIL import Image

DATA_DIR = "tree_images"
CORRUPTED_DIR = "corrupted_flagged"


def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()  # checks file integrity without fully decoding
        # verify() can leave the file object in a weird state for
        # subsequent use, so re-open and do a real decode as a second check
        with Image.open(filepath) as img:
            img.convert("RGB")
        return True
    except Exception:
        return False


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"No {DATA_DIR} directory found.")
        return

    total_checked = 0
    total_bad = 0
    bad_by_species = {}

    for species_folder in sorted(os.listdir(DATA_DIR)):
        species_path = os.path.join(DATA_DIR, species_folder)
        if not os.path.isdir(species_path):
            continue
        for fname in os.listdir(species_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            filepath = os.path.join(species_path, fname)
            total_checked += 1
            if not is_valid_image(filepath):
                dest_dir = os.path.join(CORRUPTED_DIR, species_folder)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.move(filepath, os.path.join(dest_dir, fname))
                total_bad += 1
                bad_by_species[species_folder] = bad_by_species.get(species_folder, 0) + 1

    print(f"Checked {total_checked} images.")
    print(f"Moved {total_bad} invalid files to {CORRUPTED_DIR}/\n")
    if bad_by_species:
        print("Bad files by species:")
        for folder, count in sorted(bad_by_species.items(), key=lambda x: -x[1]):
            print(f"  {folder}: {count}")
        print("\nNext: run reconcile_metadata.py to sync the metadata CSV, "
              "then re-run the downloader to top species back up to target if needed.")
    else:
        print("No corrupt files found — dataset is clean.")


if __name__ == "__main__":
    main()