import requests
import csv
import json
import re
import math
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

BASE_URL = "https://otodien.vn"
LISTING_URL = "https://otodien.vn/oto"
CSV_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "otodien", "data_xe_dien.csv")

HEADERS = ['ID', 'Tên', 'Tiền (VNĐ)', "Vị trí", 'Ngày đăng', 'Người dùng','Sao','Đã bán', 'Đang bán',
           'Thông tin mô tả', 'Tính năng nổi bật', 'Tính năng khác',
           'Kiểu dáng', 'Màu bên ngoài', 'Chiều dài(mm)', 'Chiều dài cơ sở(mm)', 'Chiều rộng(mm)',
           'khoảng sáng gầm(mm)', 'Số chỗ ngồi', 'Trọng lượng bản thân (kg)', 'Trọng lượng toàn tải (kg)',
           'Dung tích khoang hành lý (lít)', 'Dung tích khoang hành lý khi gập ghế sau (lít)',
           'Công suất tốt đa(hp)', 'Tốc độ tối đa (km/h)', '0-100(s)', 'Tầm hoạt động (km)',
           'Dung lượng pin (kWh)','Chi phí sạc đầy (VNĐ)', 'Chi phí sạc hàng tháng (VNĐ)', 'Sạc chậm (giờ)',
           'Sạc tiêu chuẩn (giờ)', 'Phần trăm sạc treo tường', 'phút sạc treo tường']

REQ_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
    'Cache-Control': 'no-cache',
    'Pragma': 'no-cache',
}

def extract_json_from_html(html, pattern=r'"items":(\[.*?\]),"statistics"'):
    matches = re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"]\)', html)
    for m in matches:
        try:
            text = m.encode('utf-8').decode('unicode_escape')
            match = re.search(pattern, text)
            if match:
                return json.loads(match.group(1))
        except Exception:
            pass
    return None

def extract_detail_json_from_html(html):
    matches = re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"]\)', html)
    for m in matches:
        try:
            text = m.encode('utf-8').decode('unicode_escape')
            match = re.search(r'"items":(\{.*?\})(?:,"others"|,"similar"|\})', text)
            if match:
                return json.loads(match.group(1))
        except Exception:
            pass
    return None

def get_car_links(page):
    page_url = f"{LISTING_URL}?page={page}"
    response = requests.get(page_url, headers=REQ_HEADERS, timeout=15)
    if response.status_code != 200:
        return None, None
        
    items = extract_json_from_html(response.text, r'"items":(\[.*?\]),"statistics"')
    if not items:
        return None, None
        
    # Also extract lastPage
    last_page = 1
    for m in re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"]\)', response.text):
        try:
            text = m.encode('utf-8').decode('unicode_escape')
            match = re.search(r'"lastPage":(\d+)', text)
            if match:
                last_page = int(match.group(1))
                break
        except Exception:
            pass
            
    links = []
    for item in items:
        detail_url = item.get('detail_url')
        if detail_url:
            links.append(BASE_URL + detail_url)
            
    return links, last_page

def extract_data(link: str):
    response = requests.get(link, headers=REQ_HEADERS, timeout=15)
    if response.status_code != 200:
        return None
        
    item = extract_detail_json_from_html(response.text)
    if not item:
        return None

    row = []
    row.append(link) # 0
    title = item.get('title', '')
    title_split = title.split('–')
    if len(title_split) > 1:
        row.append(title_split[0].strip())
    else:
        row.append(title.split('-')[0].strip())

    row.append(re.sub(r'[^\d.]', '', str(item.get('price') or ''))) # 2
    row.append(str(item.get('province', ''))) # 3
    date_str = item.get('date', '')
    if date_str:
        try:
            date_obj = datetime.strptime(date_str, "%d/%m/%Y")
            row.append(date_obj.strftime("%Y-%m-%d"))
        except:
            row.append(date_str)
    else:
        row.append("")
    row.append(str(item.get('shop_name') or item.get('name', ''))) # 5
    row.append("") # 6
    row.append(str(item.get('sold_ev', 0))) # 7
    row.append(str(item.get('publish_ev', 0))) # 8
    desc = item.get('description', '')
    if isinstance(desc, str):
        row.append(desc.replace('\n', '-').replace('\r', ''))
    else:
        row.append("")
    hl = item.get('features_highlight', [])
    hl_str = " - ".join([f.get('name', '') for f in hl if isinstance(f, dict)])
    row.append(hl_str)
    fn = item.get('features_normal', [])
    fn_more = item.get('features_normal_more', [])
    fn_str = " - ".join((fn if isinstance(fn, list) else []) + (fn_more if isinstance(fn_more, list) else []))
    row.append(fn_str)
    row.append(str(item.get('body_name', '')))
    row.append(str(item.get('color_name', '')))
    row.append(str(item.get('size_length', '')))
    row.append(str(item.get('size_wheelbase', '')))
    row.append(str(item.get('size_width', '')))
    row.append(str(item.get('size_ground_clearance', '')))
    row.append(str(item.get('seat', '')))
    row.append(str(item.get('weight_body', '')))
    row.append(str(item.get('weight_total', '')))
    row.append(str(item.get('weight_luggage', '')))
    row.append(str(item.get('weight_luggage_seat', '')))
    row.append(str(item.get('power', '')))
    row.append(str(item.get('speed', '')))
    row.append(str(item.get('speed_up', '')))
    row.append(str(item.get('range', '')))
    row.append(str(item.get('battery', '')))
    row.append(str(item.get('charging_cost', '')))
    row.append(str(item.get('charging_cost_month', '')))
    row.append(str(item.get('slow_charging', '')).replace('giờ', '').strip())
    row.append(str(item.get('standard_charging', '')).replace('giờ', '').strip())
    
    wall = str(item.get('wall_charging', ''))
    wall_nums = re.findall(r'\d+', wall)
    if len(wall_nums) >= 2:
        row.append(wall_nums[0])
        row.append(wall_nums[1])
    else:
        row.append("")
        row.append("")
    return row

def main():
    print("Step 1: Finding max pages...")
    _, last_page = get_car_links(1)
    if not last_page:
        last_page = 110
        
    print(f"Total pages: {last_page}. Extracting all links concurrently...")
    all_links = set()
    pages = list(range(1, last_page + 1))
    
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(get_car_links, p): p for p in pages}
        for future in tqdm(as_completed(futures), total=len(pages), desc="Fetching Pages"):
            try:
                links, _ = future.result()
                if links:
                    for l in links:
                        all_links.add(l)
            except Exception as e:
                pass
                
    all_links = list(all_links)
    print(f"\nFound {len(all_links)} unique cars. Extracting details concurrently...")
    
    all_data = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(extract_data, link): link for link in all_links}
        for future in tqdm(as_completed(futures), total=len(all_links), desc="Scraping Cars"):
            try:
                car_data = future.result()
                if car_data:
                    all_data.append(car_data)
            except Exception as e:
                pass

    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(HEADERS)
        for row in all_data:
            writer.writerow(row)

    print(f"\nDone! Extracted {len(all_data)} cars. Saved to {CSV_FILE}")

if __name__ == "__main__":
    main()