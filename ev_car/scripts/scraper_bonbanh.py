import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def get_value_exact(soup, label_name):
    rows = soup.find_all('div', class_=['row', 'row_last'])
    for row in rows:
        label_div = row.find('div', class_='label')
        if label_div and label_name in label_div.text:
            input_div = row.find('div', class_=['txt_input', 'inputbox'])
            if input_div:
                return input_div.text.strip()
    return "N/A"

def get_car_details(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://bonbanh.com/"
        }
        # Increased timeout slightly for concurrent connections
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.content, "html.parser")

        title_tag = soup.find("h1")
        full_title = title_tag.text.strip() if title_tag else "N/A"
        clean_title = re.sub(r'[-–—−]', '-', full_title) 
        
        ten_xe = clean_title
        gia = "N/A"
        if "-" in clean_title:
            parts = clean_title.rsplit("-", 1) 
            ten_xe = parts[0].strip()
            gia = parts[1].strip()

        notes_div = soup.find('div', class_='notes')
        ngay_dang = "N/A"
        if notes_div:
            notes_text = notes_div.text.strip()
            if "Đăng ngày" in notes_text:
                ngay_dang = notes_text.split('.')[0].replace("Đăng ngày", "").strip()

        des_div = soup.find('div', class_='des_txt')
        mo_ta = des_div.get_text(separator=" | ", strip=True) if des_div else "N/A"
        
        cname_tag = soup.find(class_='cname')
        ten_nguoi_ban = cname_tag.text.strip() if cname_tag else "N/A"

        dia_chi = "N/A"
        contact_div = soup.find('div', class_='contact-txt')
        if contact_div:
            raw_text = contact_div.get_text(separator=" ", strip=True)
            match = re.search(r'Địa chỉ\s*:\s*(.*?)(?:Website|Email|$)', raw_text, re.IGNORECASE)
            if match:
                dia_chi = match.group(1).strip()
                if not dia_chi: 
                    dia_chi = "N/A"

        car_info = {
            "Tên xe": ten_xe,
            "Giá": gia,
            "Link": url,
            "Ngày đăng": ngay_dang,
            "Tên người bán": ten_nguoi_ban, 
            "Địa chỉ": dia_chi,             
            "Mô tả": mo_ta,                 
            "Năm sản xuất": get_value_exact(soup, "Năm sản xuất"),
            "Tình trạng": get_value_exact(soup, "Tình trạng"),
            "Số Km đã đi": get_value_exact(soup, "Số Km đã đi"),
            "Xuất xứ": get_value_exact(soup, "Xuất xứ"),
            "Kiểu dáng": get_value_exact(soup, "Kiểu dáng"),
            "Hộp số": get_value_exact(soup, "Hộp số"),
            "Động cơ": get_value_exact(soup, "Động cơ"),
            "Màu ngoại thất": get_value_exact(soup, "Màu ngoại thất"),
            "Màu nội thất": get_value_exact(soup, "Màu nội thất"),
            "Số chỗ ngồi": get_value_exact(soup, "Số chỗ ngồi"),
            "Số cửa": get_value_exact(soup, "Số cửa"),
            "Dẫn động": get_value_exact(soup, "Dẫn động")
        }
        return car_info
    except Exception as e:
        return None

def fetch_page_links(page):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://bonbanh.com/"
    }
    url = "https://bonbanh.com/oto-xe-dien" if page == 1 else f"https://bonbanh.com/oto-xe-dien/page,{page}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
        soup = BeautifulSoup(response.content, "html.parser")
        car_items = soup.find_all("li", class_="car-item")
        
        links = []
        for item in car_items:
            link_tag = item.find("a")
            if link_tag and 'href' in link_tag.attrs:
                links.append("https://bonbanh.com/" + link_tag['href'].lstrip('/'))
        return links
    except:
        return []

def main():
    print("Step 1: Quét tất cả các trang gom link (Chế độ ĐA LUỒNG CỰC ĐẠI)...")
    all_links = set()
    
    # Bonbanh pagination has ~141 pages. We can query up to 150 pages in parallel.
    pages = list(range(1, 160))
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(fetch_page_links, p): p for p in pages}
        for future in tqdm(as_completed(futures), total=len(pages), desc="Fetching Pages"):
            links = future.result()
            for l in links:
                all_links.add(l)

    all_links = list(all_links)
    print(f"\n=> TỔNG CỘNG: Gom được {len(all_links)} link xe điện từ tất cả các trang.")
    
    print("\nStep 2: Bắt đầu lấy thông số chi tiết (Max workers: 50)...")
    all_cars_data = []
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(get_car_details, link): link for link in all_links}
        for future in tqdm(as_completed(futures), total=len(all_links), desc="Scraping Details"):
            res = future.result()
            if res:
                all_cars_data.append(res)

    if all_cars_data:
        df = pd.DataFrame(all_cars_data)
        cols = ["Ngày đăng", "Tên xe", "Giá", "Tên người bán", "Địa chỉ", "Năm sản xuất", "Tình trạng", "Số Km đã đi", "Xuất xứ", "Kiểu dáng", "Hộp số", "Động cơ", "Màu ngoại thất", "Màu nội thất", "Số chỗ ngồi", "Số cửa", "Dẫn động", "Mô tả", "Link"]
        df = df[cols]
        
        out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "bonbanh.csv")
        
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n XONG! Đã quét sạch {len(all_cars_data)} xe. Đã lưu tại {out_path}")

if __name__ == "__main__":
    main()
