"""
Downloads a diverse "not a tree" negative class for the model — reuses the
same validated-download pattern as download_tree_images_v6.py (checks
Content-Type + PIL can actually open the bytes before saving), just aimed
at generic non-tree categories instead of species names.

DIVERSITY MATTERS MORE THAN VOLUME HERE: a negative class of only "dog"
photos would teach the model to reject dogs specifically, not to reject
"anything that isn't one of our 10 trees" in general. The category list
below spans animals, vehicles, people, objects, buildings, and non-tree
plants (crops/flowers/grass — realistic accidental farmer photos) so the
model learns a genuinely general "not a tree" concept.

Run this on your own machine (needs internet access).
Requires: pip install requests pillow
Optional: set FLICKR_API_KEY as an environment variable for the Flickr source

Usage:
    python download_negative_images.py
"""

import os
import csv
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
from io import BytesIO

TARGET_TOTAL = 300
IMAGES_PER_CATEGORY = 12
OUTPUT_DIR = os.path.join("tree_images", "OTHER_NotATree")
METADATA_CSV = "image_metadata_tier1.csv"
ACCEPTABLE_LICENSES_TEXT = ("cc", "public domain")  # loose check for Commons LicenseShortName
FLICKR_API_KEY = os.environ.get("FLICKR_API_KEY", "")
FLICKR_CC_LICENSES = "1,2,3,4,5,6,7,8,9,10"
FIELDNAMES = ["image_id", "species_id", "scientific_name", "plant_part",
              "image_source", "license", "region_state",
              "capture_quality_flag", "source_tier", "local_filepath"]
NEGATIVE_CLASS_ID = "OTHER"

# Deliberately diverse — animals, vehicles, people, objects, buildings,
# AND non-tree plants (a farmer might well photograph a crop field by
# mistake, so the model needs to know that's not a tree either).
CATEGORIES = [
    "dog", "cat", "cow", "goat", "chicken", "person face portrait",
    "car", "motorcycle", "bicycle", "tractor",
    "house building", "brick wall", "chair furniture", "kitchen utensils",
    "food plate meal", "mobile phone", "shoe", "backpack",
    "road street", "sky clouds", "mountain landscape", "river water",
    "wheat field crop", "flower garden", "vegetable field", "grass lawn",
    "book", "laptop computer", "market vegetables stall",
]


def make_session():
    session = requests.Session()
    retry = Retry(total=5, connect=5, read=5, backoff_factor=2,
                   status_forcelist=[429, 500, 502, 503, 504],
                   allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "CarbonCreditTreeClassifier/1.0 (student research project; "
                       "contact: your-email@example.com)"
    })
    return session


def fetch_and_validate_image(session, img_url, timeout=30):
    resp = session.get(img_url, timeout=timeout)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        return None
    try:
        with Image.open(BytesIO(resp.content)) as img:
            img.verify()
    except Exception:
        return None
    return resp.content


def load_existing_progress():
    existing_ids = set()
    count = 0
    if os.path.exists(METADATA_CSV):
        with open(METADATA_CSV, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["species_id"] == NEGATIVE_CLASS_ID:
                    existing_ids.add(row["image_id"])
                    count += 1
    return existing_ids, count


def commons_search_files(session, query, limit=30):
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query", "list": "search", "srnamespace": 6,
        "srsearch": query, "srlimit": limit, "format": "json",
    }
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return [r["title"] for r in resp.json().get("query", {}).get("search", [])]


def commons_get_image_info(session, file_title):
    url = "https://commons.wikimedia.org/w/api.php"
    params = {"action": "query", "titles": file_title, "prop": "imageinfo",
              "iiprop": "url|extmetadata", "format": "json"}
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        infos = page.get("imageinfo")
        if infos:
            return infos[0]
    return None


def fetch_from_commons(session, category, needed, seen_ids, writer):
    saved = 0
    try:
        titles = commons_search_files(session, category)
    except requests.RequestException as e:
        print(f"    [ERROR] Commons search for '{category}': {e}")
        return 0
    for title in titles:
        if saved >= needed:
            break
        if not title.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        try:
            info = commons_get_image_info(session, title)
        except requests.RequestException:
            continue
        if not info or "url" not in info:
            continue
        license_short = info.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "").lower()
        if not any(tag in license_short for tag in ACCEPTABLE_LICENSES_TEXT):
            continue
        image_id = f"{NEGATIVE_CLASS_ID}_commons_{abs(hash(title)) % 10**8}"
        if image_id in seen_ids:
            continue
        try:
            img_data = fetch_and_validate_image(session, info["url"])
        except requests.RequestException:
            continue
        if img_data is None:
            continue
        filepath = os.path.join(OUTPUT_DIR, f"{image_id}.jpg")
        with open(filepath, "wb") as f:
            f.write(img_data)
        writer.writerow({
            "image_id": image_id, "species_id": NEGATIVE_CLASS_ID,
            "scientific_name": "NOT_A_TREE", "plant_part": "n/a",
            "image_source": "wikimedia_commons", "license": license_short,
            "region_state": "unknown", "capture_quality_flag": "unlabeled",
            "source_tier": f"negative_class_{category}", "local_filepath": filepath,
        })
        seen_ids.add(image_id)
        saved += 1
        time.sleep(0.3)
    return saved


def fetch_from_flickr(session, category, needed, seen_ids, writer):
    if not FLICKR_API_KEY:
        return 0
    url = "https://api.flickr.com/services/rest/"
    params = {
        "method": "flickr.photos.search", "api_key": FLICKR_API_KEY,
        "text": category, "license": FLICKR_CC_LICENSES,
        "format": "json", "nojsoncallback": 1, "per_page": needed * 2,
        "extras": "url_l,license",
    }
    try:
        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        photos = resp.json().get("photos", {}).get("photo", [])
    except (requests.RequestException, ValueError):
        return 0
    saved = 0
    for photo in photos:
        if saved >= needed:
            break
        img_url = photo.get("url_l")
        if not img_url:
            continue
        image_id = f"{NEGATIVE_CLASS_ID}_flickr_{photo['id']}"
        if image_id in seen_ids:
            continue
        try:
            img_data = fetch_and_validate_image(session, img_url)
        except requests.RequestException:
            continue
        if img_data is None:
            continue
        filepath = os.path.join(OUTPUT_DIR, f"{image_id}.jpg")
        with open(filepath, "wb") as f:
            f.write(img_data)
        writer.writerow({
            "image_id": image_id, "species_id": NEGATIVE_CLASS_ID,
            "scientific_name": "NOT_A_TREE", "plant_part": "n/a",
            "image_source": "flickr", "license": str(photo.get("license", "")),
            "region_state": "unknown", "capture_quality_flag": "unlabeled",
            "source_tier": f"negative_class_{category}", "local_filepath": filepath,
        })
        seen_ids.add(image_id)
        saved += 1
        time.sleep(0.3)
    return saved


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session = make_session()
    seen_ids, already_have = load_existing_progress()
    print(f"Already have {already_have} negative-class images.\n")

    if not FLICKR_API_KEY:
        print("Note: FLICKR_API_KEY not set — Flickr source will be skipped for this run.\n")

    file_is_new = not os.path.exists(METADATA_CSV)
    with open(METADATA_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if file_is_new:
            writer.writeheader()

        total_new = 0
        for category in CATEGORIES:
            if already_have + total_new >= TARGET_TOTAL:
                print("Reached target total — stopping early.")
                break
            print(f"Category: {category}")
            got = fetch_from_commons(session, category, IMAGES_PER_CATEGORY, seen_ids, writer)
            if got < IMAGES_PER_CATEGORY:
                got += fetch_from_flickr(session, category, IMAGES_PER_CATEGORY - got, seen_ids, writer)
            print(f"  +{got} images")
            total_new += got
            f.flush()

    print(f"\nTotal negative-class images now: {already_have + total_new}")
    print(f"Saved under: {OUTPUT_DIR}/")
    print("\nNext: manually skim this folder like you did for the tree species — "
          "remove anything mislabeled or low quality — then re-run training. "
          "The new 11th class will be picked up automatically since the training "
          "script builds its class list from folder names.")


if __name__ == "__main__":
    main()