"""
Chotot.com Electric Motorcycle Scraper
======================================
Scrapes all electric motorcycle (xe máy điện) listings from Chotot Gateway API.
API: https://gateway.chotot.com/v1/public/ad-listing
Filter: cg=2020 (Xe máy) + motorbiketype=4 (Xe máy điện)
"""

import requests
import json
import time
import os
from datetime import datetime


API_URL = "https://gateway.chotot.com/v1/public/ad-listing"
PARAMS_BASE = {
    "cg": 2020,
    "motorbiketype": 4,
    "limit": 50,
    "key_param_included": "true",
}
OUTPUT_DIR = r"c:\Ki_5\final\ev_car\data\raw\chotot"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "motorbikes.json")
DELAY = 0.5  # seconds between API calls


def transform_ad(ad):
    """Transform a raw API ad object into our standardized format."""
    # Safely get list_time and convert to datetime
    list_time_ms = ad.get("list_time")
    if list_time_ms:
        exact_date = datetime.fromtimestamp(list_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
    else:
        exact_date = ""

    # Extract motorbiketype from params array
    motorbiketype_value = ""
    for p in ad.get("params", []):
        if p.get("id") == "motorbiketype":
            motorbiketype_value = p.get("value", "")
            break

    # Build address
    area_name = ad.get("area_name", "")
    region_name = ad.get("region_name", "")
    address = f"{area_name}, {region_name}" if area_name or region_name else ""

    return {
        "url": f"https://xe.chotot.com/mua-ban-xe-may/{ad.get('list_id', '')}.htm",
        "exact_date_posted": exact_date,
        "ad": {
            "subject": ad.get("subject", ""),
            "price": ad.get("price", 0),
            "account_name": ad.get("account_name", ""),
            "body": ad.get("body", ""),
        },
        "ad_params": {
            "address": {"value": address},
            "mfdate": {"value": str(ad.get("regdate", ""))},
            "condition_ad": {"value": ad.get("condition_ad_name", "")},
            "mileage_v2": {"value": str(ad.get("mileage_v2", ""))},
            "fuel": {"value": "Điện"},
            "motorbiketype": {"value": motorbiketype_value},
        },
        "vehicle_type": "xe_may_dien",
    }


def main():
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Starting electric motorbike scraper...")
    print(f"API: {API_URL}")
    print(f"Filters: cg=2020, motorbiketype=4")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    })

    all_records = []
    offset = 0
    total = None

    while True:
        params = {**PARAMS_BASE, "o": offset}

        try:
            resp = session.get(API_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  [ERROR] Request failed at offset {offset}: {e}")
            # Retry once after a longer delay
            time.sleep(3)
            try:
                resp = session.get(API_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e2:
                print(f"  [FATAL] Retry also failed at offset {offset}: {e2}")
                break

        ads = data.get("ads", [])
        if total is None:
            total = data.get("total", 0)
            print(f"  Total listings reported by API: {total}")

        if not ads:
            print(f"  No more ads at offset {offset}. Stopping.")
            break

        for ad in ads:
            record = transform_ad(ad)
            all_records.append(record)

        count = len(all_records)
        if count % 100 < len(ads) or offset == 0:
            print(f"  Collected {count}/{total} records (offset={offset})")

        offset += len(ads)

        # Safety check: don't go past the total
        if total and offset >= total:
            print(f"  Reached total ({total}). Stopping.")
            break

        time.sleep(DELAY)

    # Save output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8-sig") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] Done!")
    print(f"  Total records saved: {len(all_records)}")
    print(f"  Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
