"""
Run this after manually deleting bad images from tree_images/<folder>/.
It removes metadata rows pointing to files that no longer exist, and
reports per-species counts so you know which species are running thin
before training.

Usage:
    python reconcile_metadata.py
"""

import os
import csv

OUTPUT_DIR = "tree_images"
METADATA_CSV = "image_metadata_tier1.csv"
LOW_COUNT_FLOOR = 150  # heads-up threshold, scaled up for the 300/species target


def main():
    if not os.path.exists(METADATA_CSV):
        print(f"No {METADATA_CSV} found — nothing to reconcile.")
        return

    with open(METADATA_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    kept_rows = [r for r in rows if os.path.exists(r["local_filepath"])]
    removed = len(rows) - len(kept_rows)

    with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    counts = {}
    for r in kept_rows:
        key = (r["species_id"], r["scientific_name"])
        counts[key] = counts.get(key, 0) + 1

    print(f"Removed {removed} stale metadata rows (files you deleted during QA).")
    print(f"Metadata now reflects {len(kept_rows)} images across {len(counts)} species.\n")

    print("--- Post-QA counts ---")
    low = []
    for (species_id, sci_name), count in sorted(counts.items()):
        flag = ""
        if count < LOW_COUNT_FLOOR:
            flag = f"  <-- below floor of {LOW_COUNT_FLOOR}, consider topping up"
            low.append((species_id, sci_name, count))
        print(f"  {species_id} {sci_name}: {count}{flag}")

    if low:
        print(f"\n{len(low)} species below the {LOW_COUNT_FLOOR}-image floor — "
              f"fine to proceed with training, but these are your weakest classes.")
    else:
        print(f"\nAll species at or above {LOW_COUNT_FLOOR} images. Good to proceed to training.")


if __name__ == "__main__":
    main()