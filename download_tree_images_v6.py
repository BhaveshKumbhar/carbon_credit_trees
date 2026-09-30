"""
v5: Adds two new image sources on top of v4's iNaturalist + Wikimedia Commons,
plus automatic blur detection.

NEW SOURCES:
- GBIF (Global Biodiversity Information Facility) — aggregates iNaturalist,
  herbarium specimens, and India Biodiversity Portal (India-specific,
  Creative-Commons-by-policy). No API key needed. Genuinely expands the
  pool beyond what iNaturalist alone offers, since it includes providers
  iNaturalist doesn't (e.g. IBP).
- Flickr — huge photo volume with real CC-license filtering. Requires a
  free API key (get one in ~30 seconds at https://www.flickr.com/services/apps/create/apply/).
  Set it as an environment variable: FLICKR_API_KEY=your_key_here
  If not set, Flickr is skipped automatically (not an error).

NEW: automatic blur detection (Laplacian variance) flags likely-blurry
images into a separate folder for review instead of silently including
them — addresses "not properly focused" without needing a new source.

Source order per species: iNaturalist tiers -> GBIF -> Wikimedia Commons
-> Flickr (if key set). Each only runs if still short of target.

Run this on your own machine (needs internet access).
Requires: pip install requests
Optional (for blur detection): pip install opencv-python-headless

Usage:
    python download_tree_images_v5.py
"""

import os
import csv
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image

TARGET_IMAGES_PER_SPECIES = 300
INDIA_PLACE_ID = 6903          # iNaturalist
INDIA_COUNTRY_CODE = "IN"      # GBIF
OUTPUT_DIR = "tree_images"
BLUR_FLAGGED_DIR = "tree_images_blur_flagged"
METADATA_CSV = "image_metadata_tier1.csv"
ACCEPTABLE_LICENSES = {"cc0", "cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa"}
MAX_PAGES_PER_TIER = 12
COMMONS_SEARCH_LIMIT = 200
GBIF_PAGE_SIZE = 300
#FLICKR_API_KEY = os.environ.get("FLICKR_API_KEY", "")
BLUR_VARIANCE_THRESHOLD = 100.0  # lower = more permissive; tune if needed
FIELDNAMES = ["image_id", "species_id", "scientific_name", "plant_part",
              "image_source", "license", "region_state",
              "capture_quality_flag", "source_tier", "local_filepath"]

SPECIES_LIST = [
    ("SP001", "Eucalyptus", "Eucalyptus tereticornis"),
    ("SP002", "Subabul", "Leucaena leucocephala"),
    ("SP003", "Shisham", "Dalbergia sissoo"),
    ("SP004", "Poplar", "Populus deltoides"),
    ("SP005", "Casuarina", "Casuarina equisetifolia"),
    ("SP007", "Teak", "Tectona grandis"),
    ("SP008", "Bamboo", "Bambusa bambos"),
    ("SP013", "Mango", "Mangifera indica"),
    ("SP021", "Jackfruit", "Artocarpus heterophyllus"),
    ("SP027", "Neem", "Azadirachta indica"),
]

INAT_TIERS = [
    {"name": "tier1_india_research", "place_id": INDIA_PLACE_ID, "quality_grade": "research"},
    {"name": "tier2_india_any", "place_id": INDIA_PLACE_ID, "quality_grade": "research,needs_id"},
    {"name": "tier3_global_research", "place_id": None, "quality_grade": "research"},
    {"name": "tier4_global_any", "place_id": None, "quality_grade": "research,needs_id"},
]


def make_session():
    session = requests.Session()
    retry = Retry(total=5, connect=5, read=5, backoff_factor=2,
                   status_forcelist=[429, 500, 502, 503, 504],
                   allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    # Wikimedia (and good practice generally) requires a real User-Agent
    # identifying the client — without this, Wikimedia returns an HTML
    # "robot policy" rejection page instead of the image, which the old
    # script was saving as if it were a valid .jpg. See:
    # https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy
    session.headers.update({
        "User-Agent": "CarbonCreditTreeClassifier/1.0 (student research project; "
                       "contact: bhavesh.kumbhar241@vit.edu)"
    })
    return session


def fetch_and_validate_image(session, img_url, timeout=30):
    """Downloads a URL and returns the bytes ONLY if they're a genuine,
    openable image — never trust an HTTP 200 status alone, since APIs
    can return HTML error/rejection pages with a 200 status. Returns
    None if the content isn't a real image, so callers can skip it
    instead of saving garbage to disk."""
    resp = session.get(img_url, timeout=timeout)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        return None  # e.g. text/html rejection page
    try:
        from io import BytesIO
        with Image.open(BytesIO(resp.content)) as img:
            img.verify()
    except Exception:
        return None  # bytes claimed to be an image but PIL can't open them
    return resp.content


def load_existing_progress():
    existing_rows = []
    counts = {}
    if os.path.exists(METADATA_CSV):
        with open(METADATA_CSV, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_rows.append(row)
                counts[row["species_id"]] = counts.get(row["species_id"], 0) + 1
    return existing_rows, counts


# def check_blur(filepath):
#     """Returns True if the image looks blurry. Silently returns False
#     (assume sharp) if opencv isn't installed, so this is fully optional."""
#     try:
#         import cv2
#     except ImportError:
#         return False
#     img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         return True  # unreadable/corrupt — treat as bad
#     variance = cv2.Laplacian(img, cv2.CV_64F).var()
#     return variance < BLUR_VARIANCE_THRESHOLD


# def maybe_flag_blur(filepath, species_folder_name):
#     if check_blur(filepath):
#         flagged_dir = os.path.join(BLUR_FLAGGED_DIR, species_folder_name)
#         os.makedirs(flagged_dir, exist_ok=True)
#         new_path = os.path.join(flagged_dir, os.path.basename(filepath))
#         os.replace(filepath, new_path)
#         return new_path  # caller should NOT count this as a kept image
#     return None


# ---------- iNaturalist (same as v4) ----------

def inat_fetch(session, scientific_name, tier, page):
    url = "https://api.inaturalist.org/v1/observations"
    params = {
        "taxon_name": scientific_name, "quality_grade": tier["quality_grade"],
        "photos": "true", "photo_licensed": "true", "per_page": 30,
        "page": page, "order_by": "votes",
    }
    if tier["place_id"]:
        params["place_id"] = tier["place_id"]
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("results", [])


def try_inaturalist(session, species_id, scientific_name, species_dir, folder_name,
                     needed, seen_image_ids, writer):
    saved = 0
    for tier in INAT_TIERS:
        if saved >= needed:
            break
        for page in range(1, MAX_PAGES_PER_TIER + 1):
            if saved >= needed:
                break
            try:
                observations = inat_fetch(session, scientific_name, tier, page)
            except requests.RequestException as e:
                print(f"    [ERROR] iNat {tier['name']} page {page}: {e}")
                continue
            if not observations:
                break
            for obs in observations:
                if saved >= needed:
                    break
                for photo in obs.get("photos", []):
                    if saved >= needed:
                        break
                    license_code = (photo.get("license_code") or "").lower()
                    if license_code not in ACCEPTABLE_LICENSES:
                        continue
                    image_id = f"{species_id}_inat_{photo['id']}"
                    if image_id in seen_image_ids:
                        continue
                    img_url = photo["url"].replace("square", "large")
                    filepath = os.path.join(species_dir, f"{image_id}.jpg")
                    try:
                        img_data = fetch_and_validate_image(session, img_url)
                    except requests.RequestException as e:
                        print(f"    [skip] {image_id}: {e}")
                        continue
                    if img_data is None:
                        continue  # not a real image (e.g. rejection page) — skip silently
                    with open(filepath, "wb") as f:
                        f.write(img_data)
                    # if maybe_flag_blur(filepath, folder_name):
                    #     continue  # moved to blur-flagged folder, don't count
                    writer.writerow({
                        "image_id": image_id, "species_id": species_id,
                        "scientific_name": scientific_name, "plant_part": "unlabeled",
                        "image_source": "inaturalist", "license": license_code,
                        "region_state": obs.get("place_guess", "unknown"),
                        "capture_quality_flag": "unlabeled",
                        "source_tier": tier["name"], "local_filepath": filepath,
                    })
                    seen_image_ids.add(image_id)
                    saved += 1
                    time.sleep(0.5)
    return saved


# ---------- GBIF (new) ----------

def gbif_fetch(session, scientific_name, country, offset):
    url = "https://api.gbif.org/v1/occurrence/search"
    params = {
        "scientificName": scientific_name, "mediaType": "StillImage",
        "limit": GBIF_PAGE_SIZE, "offset": offset,
    }
    if country:
        params["country"] = country
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def try_gbif(session, species_id, scientific_name, species_dir, folder_name,
             needed, seen_image_ids, writer):
    saved = 0
    for country in [INDIA_COUNTRY_CODE, None]:  # India first, then global
        if saved >= needed:
            break
        offset = 0
        while saved < needed:
            try:
                data = gbif_fetch(session, scientific_name, country, offset)
            except requests.RequestException as e:
                print(f"    [ERROR] GBIF (country={country}) offset {offset}: {e}")
                break
            results = data.get("results", [])
            if not results:
                break
            for occ in results:
                if saved >= needed:
                    break
                license_raw = (occ.get("license") or "").upper()
                if not license_raw.startswith("CC") and license_raw != "CC0_1_0":
                    continue
                for media in occ.get("media", []):
                    if saved >= needed:
                        break
                    if media.get("type") != "StillImage":
                        continue
                    img_url = media.get("identifier")
                    if not img_url:
                        continue
                    image_id = f"{species_id}_gbif_{occ.get('key')}"
                    if image_id in seen_image_ids:
                        continue
                    filepath = os.path.join(species_dir, f"{image_id}.jpg")
                    try:
                        img_data = fetch_and_validate_image(session, img_url)
                    except requests.RequestException as e:
                        print(f"    [skip] {image_id}: {e}")
                        continue
                    if img_data is None:
                        continue
                    with open(filepath, "wb") as f:
                        f.write(img_data)
                    # if maybe_flag_blur(filepath, folder_name):
                    #     continue
                    writer.writerow({
                        "image_id": image_id, "species_id": species_id,
                        "scientific_name": scientific_name, "plant_part": "unlabeled",
                        "image_source": "gbif", "license": license_raw,
                        "region_state": occ.get("country", "unknown"),
                        "capture_quality_flag": "unlabeled",
                        "source_tier": f"gbif_{'india' if country else 'global'}",
                        "local_filepath": filepath,
                    })
                    seen_image_ids.add(image_id)
                    saved += 1
                    time.sleep(0.3)
            offset += len(results)
            if data.get("endOfRecords", True):
                break
    return saved


# ---------- Wikimedia Commons (same as v4) ----------

def commons_search_files(session, scientific_name, limit=COMMONS_SEARCH_LIMIT):
    url = "https://commons.wikimedia.org/w/api.php"
    all_titles = []
    sroffset = 0
    while len(all_titles) < limit:
        params = {
            "action": "query", "list": "search", "srnamespace": 6,
            "srsearch": f'"{scientific_name}" tree', "srlimit": min(50, limit - len(all_titles)),
            "sroffset": sroffset, "format": "json",
        }
        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("query", {}).get("search", [])
        if not results:
            break
        all_titles.extend([r["title"] for r in results])
        sroffset += len(results)
        if "continue" not in data:
            break
    return all_titles


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


def try_commons(session, species_id, scientific_name, species_dir, folder_name,
                 needed, seen_image_ids, writer):
    saved = 0
    try:
        titles = commons_search_files(session, scientific_name)
    except requests.RequestException as e:
        print(f"    [ERROR] Commons search failed: {e}")
        return 0
    for title in titles:
        if saved >= needed:
            break
        if not title.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        try:
            info = commons_get_image_info(session, title)
        except requests.RequestException as e:
            print(f"    [ERROR] Commons imageinfo failed for {title}: {e}")
            continue
        if not info or "url" not in info:
            continue
        license_short = info.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "")
        if "CC" not in license_short and "public domain" not in license_short.lower():
            continue
        image_id = f"{species_id}_commons_{abs(hash(title)) % 10**8}"
        if image_id in seen_image_ids:
            continue
        filepath = os.path.join(species_dir, f"{image_id}.jpg")
        try:
            img_data = fetch_and_validate_image(session, info["url"])
        except requests.RequestException as e:
            print(f"    [skip] {image_id}: {e}")
            continue
        if img_data is None:
            continue
        with open(filepath, "wb") as f:
            f.write(img_data)
        # if maybe_flag_blur(filepath, folder_name):
        #     continue
        writer.writerow({
            "image_id": image_id, "species_id": species_id,
            "scientific_name": scientific_name, "plant_part": "unlabeled",
            "image_source": "wikimedia_commons", "license": license_short,
            "region_state": "unknown", "capture_quality_flag": "unlabeled",
            "source_tier": "commons_fallback", "local_filepath": filepath,
        })
        seen_image_ids.add(image_id)
        saved += 1
        time.sleep(0.3)
    return saved


# ---------- Flickr (new, optional — needs FLICKR_API_KEY) ----------

# FLICKR_CC_LICENSES = "1,2,3,4,5,6,7,8,9,10"  # everything except 0 (All Rights Reserved)

# def try_flickr(session, species_id, scientific_name, species_dir, folder_name,
#                 needed, seen_image_ids, writer):
#     if not FLICKR_API_KEY:
#         return 0  # silently skip — not an error, just not configured
#     saved = 0
#     page = 1
#     while saved < needed:
#         url = "https://api.flickr.com/services/rest/"
#         params = {
#             "method": "flickr.photos.search", "api_key": FLICKR_API_KEY,
#             "text": f"{scientific_name} tree", "license": FLICKR_CC_LICENSES,
#             "format": "json", "nojsoncallback": 1, "per_page": 250, "page": page,
#             "extras": "url_l,license",
#         }
#         try:
#             resp = session.get(url, params=params, timeout=30)
#             resp.raise_for_status()
#             data = resp.json()
#         except (requests.RequestException, ValueError) as e:
#             print(f"    [ERROR] Flickr page {page}: {e}")
#             break
#         photos = data.get("photos", {}).get("photo", [])
#         if not photos:
#             break
#         for photo in photos:
#             if saved >= needed:
#                 break
#             img_url = photo.get("url_l")
#             if not img_url:
#                 continue  # no large size available
#             image_id = f"{species_id}_flickr_{photo['id']}"
#             if image_id in seen_image_ids:
#                 continue
#             filepath = os.path.join(species_dir, f"{image_id}.jpg")
#             try:
#                 img_data = fetch_and_validate_image(session, img_url)
#             except requests.RequestException as e:
#                 print(f"    [skip] {image_id}: {e}")
#                 continue
#             if img_data is None:
#                 continue
#             with open(filepath, "wb") as f:
#                 f.write(img_data)
#             if maybe_flag_blur(filepath, folder_name):
#                 continue
#             writer.writerow({
#                 "image_id": image_id, "species_id": species_id,
#                 "scientific_name": scientific_name, "plant_part": "unlabeled",
#                 "image_source": "flickr", "license": str(photo.get("license", "")),
#                 "region_state": "unknown", "capture_quality_flag": "unlabeled",
#                 "source_tier": "flickr", "local_filepath": filepath,
#             })
#             seen_image_ids.add(image_id)
#             saved += 1
#             time.sleep(0.3)
#         page += 1
#         if page > 20:  # safety cap
#             break
#     return saved


# ---------- Main ----------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session = make_session()
    existing_rows, counts = load_existing_progress()
    seen_image_ids = {row["image_id"] for row in existing_rows}

    # if not FLICKR_API_KEY:
    #     print("Note: FLICKR_API_KEY not set — Flickr source will be skipped. "
    #           "Get a free key at https://www.flickr.com/services/apps/create/apply/ "
    #           "and set it as an environment variable to enable it.\n")

    # try:
    #     import cv2  # noqa
    #     print("Blur detection: enabled (opencv found)\n")
    # except ImportError:
    #     print("Blur detection: DISABLED — run 'pip install opencv-python-headless' to enable.\n")

    file_is_new = not os.path.exists(METADATA_CSV)
    with open(METADATA_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if file_is_new:
            writer.writeheader()

        totals = {}
        for species_id, common_name, scientific_name in SPECIES_LIST:
            already_have = counts.get(species_id, 0)
            needed = TARGET_IMAGES_PER_SPECIES - already_have
            folder_name = f"{species_id}_{common_name.replace(' ', '_')}"
            species_dir = os.path.join(OUTPUT_DIR, folder_name)
            os.makedirs(species_dir, exist_ok=True)

            print(f"\nFetching: {common_name} ({scientific_name}) — have {already_have}, need {max(needed,0)} more")
            if needed <= 0:
                print(f"  already at target — skipping")
                totals[common_name] = already_have
                continue

            got = 0
            for source_fn, source_name in [
                (try_inaturalist, "iNaturalist"),
                (try_gbif, "GBIF"),
                (try_commons, "Wikimedia Commons"),
                #(try_flickr, "Flickr"),
            ]:
                remaining = needed - got
                if remaining <= 0:
                    break
                n = source_fn(session, species_id, scientific_name, species_dir, folder_name,
                               remaining, seen_image_ids, writer)
                print(f"    {source_name}: +{n}")
                got += n

            total = already_have + got
            print(f"  {common_name}: {got} new -> {total}/{TARGET_IMAGES_PER_SPECIES}")
            totals[common_name] = total
            f.flush()

    print("\n--- Summary ---")
    shortfall = [(name, c) for name, c in totals.items() if c < TARGET_IMAGES_PER_SPECIES]
    print(f"Species still below target: {len(shortfall)}")
    for name, c in shortfall:
        print(f"  - {name}: {c} images")
    if not shortfall:
        print("All species at target!")
    #print(f"\nBlur-flagged images (if any) moved to: {BLUR_FLAGGED_DIR}/ — spot-check before discarding")


if __name__ == "__main__":
    main()