import requests
import csv
import json
import re
import math
from datetime import datetime

BASE_URL = "https://otodien.vn"
LISTING_URL = "https://otodien.vn/oto"
CSV_FILE = "data_xe_dien.csv"

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
    response = requests.get(page_url, headers=REQ_HEADERS)
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
    response = requests.get(link, headers=REQ_HEADERS)
    if response.status_code != 200:
        return None
        
    item = extract_detail_json_from_html(response.text)
    if not item:
        return None

    row = []
    
    # 0: ID
    row.append(link)

    # 1: Tên
    title = item.get('title', '')
    title_split = title.split('–')
    if len(title_split) > 1:
        row.append(title_split[0].strip())
    else:
        row.append(title.split('-')[0].strip())

    # 2: Tiền (VNĐ)
    row.append(str(item.get('origin_price', '')))

    # 3: Vị trí
    row.append(str(item.get('province', '')))

    # 4: Ngày đăng
    date_str = item.get('date', '')
    if date_str:
        try:
            date_obj = datetime.strptime(date_str, "%d/%m/%Y")
            row.append(date_obj.strftime("%Y-%m-%d"))
        except:
            row.append(date_str)
    else:
        row.append("")

    # 5: Người dùng
    row.append(str(item.get('shop_name') or item.get('name', '')))

    # 6: Sao
    row.append("") # Ko có trong json

    # 7: Đã bán
    row.append(str(item.get('sold_ev', 0)))

    # 8: Đang bán
    row.append(str(item.get('publish_ev', 0)))

    # 9: Thông tin mô tả
    # Mô tả thường bị map ra ngoài dạng $12, nên text direct có thể empty.
    desc = item.get('description', '')
    if isinstance(desc, str):
        row.append(desc.replace('\n', '-').replace('\r', ''))
    else:
        row.append("")

    # 10: Tính năng nổi bật
    hl = item.get('features_highlight', [])
    hl_str = " - ".join([f.get('name', '') for f in hl if isinstance(f, dict)])
    row.append(hl_str)

    # 11: Tính năng khác
    fn = item.get('features_normal', [])
    fn_more = item.get('features_normal_more', [])
    fn_str = " - ".join((fn if isinstance(fn, list) else []) + (fn_more if isinstance(fn_more, list) else []))
    row.append(fn_str)

    # 12: Kiểu dáng
    row.append(str(item.get('body_name', '')))
    # 13: Màu bên ngoài
    row.append(str(item.get('color_name', '')))
    # 14: Chiều dài(mm)
    row.append(str(item.get('size_length', '')))
    # 15: Chiều dài cơ sở(mm)
    row.append(str(item.get('size_wheelbase', '')))
    # 16: Chiều rộng(mm)
    row.append(str(item.get('size_width', '')))
    # 17: khoảng sáng gầm(mm)
    row.append(str(item.get('size_ground_clearance', '')))
    # 18: Số chỗ ngồi
    row.append(str(item.get('seat', '')))
    # 19: Trọng lượng bản thân (kg)
    row.append(str(item.get('weight_body', '')))
    # 20: Trọng lượng toàn tải (kg)
    row.append(str(item.get('weight_total', '')))
    # 21: Dung tích khoang hành lý (lít)
    row.append(str(item.get('weight_luggage', '')))
    # 22: Dung tích khoang hành lý khi gập ghế sau (lít)
    row.append(str(item.get('weight_luggage_seat', '')))
    # 23: Công suất tốt đa(hp)
    row.append(str(item.get('power', '')))
    # 24: Tốc độ tối đa (km/h)
    row.append(str(item.get('speed', '')))
    # 25: 0-100(s)
    row.append(str(item.get('speed_up', '')))
    # 26: Tầm hoạt động (km)
    row.append(str(item.get('range', '')))
    # 27: Dung lượng pin (kWh)
    row.append(str(item.get('battery', '')))
    # 28: Chi phí sạc đầy (VNĐ)
    row.append(str(item.get('charging_cost', '')))
    # 29: Chi phí sạc hàng tháng (VNĐ)
    row.append(str(item.get('charging_cost_month', '')))
    # 30: Sạc chậm (giờ)
    row.append(str(item.get('slow_charging', '')).replace('giờ', '').strip())
    # 31: Sạc tiêu chuẩn (giờ)
    row.append(str(item.get('standard_charging', '')).replace('giờ', '').strip())
    
    # 32: Phần trăm sạc treo tường, 33: phút sạc treo tường
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
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(HEADERS)

        all_links = []
        page = 1
        last_page = 1
        
        while page <= last_page:
            print(f"Đang quét trang danh sách {page}/{last_page}...")
            links, extracted_last_page = get_car_links(page)
            if links is None:
                break
            all_links.extend(links)
            if extracted_last_page and extracted_last_page > last_page:
                last_page = extracted_last_page
            
            page += 1

        all_links = list(set(all_links))
        print(f"Tìm thấy {len(all_links)} xe. Bắt đầu cào chi tiết...")

        for i, link in enumerate(all_links, 1):
            try:
                car_data = extract_data(link)
                if car_data:
                    writer.writerow(car_data)
                if i % 10 == 0:
                    print(f"Đã cào {i}/{len(all_links)} xe.")
            except Exception as e:
                print(f"Lỗi khi cào link {link}: {e}")

    print(f"Hoàn thành! Dữ liệu đã được lưu vào file {CSV_FILE}")

if __name__ == "__main__":
    main()