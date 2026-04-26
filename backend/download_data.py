"""
download_data.py — fetches all project data files.
Run from the socratic-ot project root:
    python3 download_data.py
"""

import os
import time
import requests
from tqdm import tqdm

CHUNK_SIZE = 1024 * 64  # 64 KB

# Wikimedia requires a descriptive User-Agent per:
# https://w.wiki/GHai  (https://meta.wikimedia.org/wiki/User-Agent_policy)
WIKIMEDIA_UA = (
    "SocraticOT-NLP-Project/1.0 "
    "(CSE 635 UB; educational non-commercial use; "
    "github.com/socratic-ot) "
    "python-requests/2.x"
)

OPENSTAX_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://openstax.org/",
    "Accept": "application/pdf,application/octet-stream,*/*",
}


def download(url: str, dest: str, label: str,
             headers: dict = None, delay: float = 0.0) -> bool:
    """Stream-download url to dest with a progress bar. Returns True on success."""
    if delay:
        time.sleep(delay)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # Skip if already downloaded and non-empty
    if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
        print(f"  –  {label}: already exists ({os.path.getsize(dest)/1e6:.1f} MB), skipping")
        return True

    _headers = headers or {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, stream=True, timeout=120,
                            headers=_headers, allow_redirects=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            desc=label,
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            ncols=90,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                bar.update(len(chunk))
        size = os.path.getsize(dest)
        print(f"  ✓  {dest}  ({size/1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ✗  {label}: {e}")
        # Remove partial file
        if os.path.exists(dest):
            os.remove(dest)
        return False


# ── File manifest ─────────────────────────────────────────────────────────────

# OpenStax CDN blocks scripted downloads; we try with a browser UA + referer.
# If these still 403, you must download manually from openstax.org (see README).
TEXTBOOKS = [
    {
        "label": "OpenStax Anatomy & Physiology 2e",
        "url": "https://assets.openstax.org/oscms-prodcms/media/documents/AnatomyandPhysiology2e-WEB_7Zesqk6.pdf",
        "dest": "data/raw/textbooks/openStax_AP2e.pdf",
        "headers": OPENSTAX_HEADERS,
    },
    {
        "label": "OpenStax University Physics Volume 1",
        "url": "https://assets.openstax.org/oscms-prodcms/media/documents/UniversityPhysicsVolume1-WEB_0YHkReF.pdf",
        "dest": "data/raw/physics/openStax_physics_v1.pdf",
        "headers": OPENSTAX_HEADERS,
    },
]

# Wikimedia Commons: use Special:FilePath which redirects to the real CDN URL.
# Requires proper User-Agent + 1-second delay between requests per their policy.
DIAGRAMS = [
    {
        "label": "brachial_plexus",
        "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Gray808.png",
        "dest": "data/raw/diagrams/grays/brachial_plexus.png",
    },
    {
        "label": "ulnar_nerve_hand",
        "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Gray811and813.PNG",
        "dest": "data/raw/diagrams/grays/ulnar_nerve_hand.png",
    },
    {
        "label": "median_nerve",
        "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Gray804.png",
        "dest": "data/raw/diagrams/grays/median_nerve.png",
    },
    {
        "label": "radial_nerve",
        "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Gray812and814.PNG",
        "dest": "data/raw/diagrams/grays/radial_nerve.png",
    },
    {
        "label": "spinal_cord_cross_section",
        "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Spinal_cord_tracts_-_English.svg",
        "dest": "data/raw/diagrams/grays/spinal_cord_cross_section.png",
    },
    {
        "label": "dermatomes_upper_limb",
        # Gray814 = cutaneous nerves of right upper extremity (dermatomes)
        "url": "https://upload.wikimedia.org/wikipedia/commons/b/b0/Gray814.png",
        "dest": "data/raw/diagrams/grays/dermatomes_upper_limb.png",
    },
    {
        "label": "rotator_cuff",
        # Gray412 = muscles of the shoulder joint, lateral view (SITS muscles)
        "url": "https://upload.wikimedia.org/wikipedia/commons/8/8f/Gray412.png",
        "dest": "data/raw/diagrams/grays/rotator_cuff.png",
    },
    {
        "label": "carpal_tunnel",
        # Gray427 = transverse section through wrist showing carpal tunnel contents
        "url": "https://upload.wikimedia.org/wikipedia/commons/3/31/Gray427.png",
        "dest": "data/raw/diagrams/grays/carpal_tunnel.png",
    },
    {
        "label": "motor_cortex_homunculus",
        "url": "https://commons.wikimedia.org/wiki/Special:FilePath/1421_Sensory_Homunculus.jpg",
        "dest": "data/raw/diagrams/grays/motor_cortex_homunculus.png",
    },
    {
        "label": "hand_intrinsic_muscles",
        "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Gray426.png",
        "dest": "data/raw/diagrams/grays/hand_intrinsic_muscles.png",
    },
]


def main():
    failed = []

    print("\n═══ TEXTBOOKS ══════════════════════════════════════════════════════")
    for f in TEXTBOOKS:
        ok = download(f["url"], f["dest"], f["label"],
                      headers=f.get("headers"))
        if not ok:
            failed.append(f["label"])

    print("\n═══ GRAY'S DIAGRAMS (Wikimedia Commons) ════════════════════════════")
    for f in DIAGRAMS:
        ok = download(f["url"], f["dest"], f["label"],
                      headers={"User-Agent": WIKIMEDIA_UA},
                      delay=1.5)   # 1.5 s between requests per Wikimedia policy
        if not ok:
            failed.append(f["label"])

    print("\n═══ SUMMARY ════════════════════════════════════════════════════════")
    total = len(TEXTBOOKS) + len(DIAGRAMS)
    passed = total - len(failed)
    print(f"  {passed}/{total} files downloaded successfully")
    if failed:
        print("  FAILED:")
        for name in failed:
            print(f"    - {name}")
    else:
        print("  All files OK — ready for verify_downloads.py")


if __name__ == "__main__":
    main()
