"""Check what JSON fields otodien.vn returns for a single car detail page"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import requests
import re
import json

BASE_URL = "https://otodien.vn"
REQ_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept-Language': 'vi-VN,vi;q=0.9',
}

# Get a sample car link from page 1
response = requests.get(f"{BASE_URL}/oto?page=1", headers=REQ_HEADERS, timeout=15)
matches = re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)', response.text)
for m in matches:
    try:
        text = m.encode('utf-8').decode('unicode_escape')
        match = re.search(r'"items":(\[.*?\]),"statistics"', text)
        if match:
            items = json.loads(match.group(1))
            if items:
                detail_url = items[0].get('detail_url', '')
                link = BASE_URL + detail_url
                print(f"Testing car: {link}")
                break
    except:
        pass

# Now fetch detail page and dump ALL JSON keys
response2 = requests.get(link, headers=REQ_HEADERS, timeout=15)
matches2 = re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)', response2.text)
for m in matches2:
    try:
        text = m.encode('utf-8').decode('unicode_escape')
        match = re.search(r'"items":(\{.*?\})(?:,"others"|,"similar"|\})', text)
        if match:
            item = json.loads(match.group(1))
            print(f"\n=== ALL JSON KEYS ({len(item)} keys) ===")
            for k, v in sorted(item.items()):
                val_str = str(v)[:80]
                print(f"  {k}: {val_str}")
            
            # Check specifically for missing fields
            print("\n=== CHECKING MISSING FIELDS ===")
            for field in ['year', 'nam', 'condition', 'tinh_trang', 'status', 
                          'mileage', 'km', 'odometer', 'origin', 'xuat_xu',
                          'interior_color', 'mau_noi_that', 'door', 'so_cua',
                          'drivetrain', 'dan_dong', 'drive', 'transmission']:
                val = item.get(field, 'NOT FOUND')
                print(f"  {field}: {val}")
            break
    except Exception as e:
        pass
