# Hướng dẫn cào dữ liệu (Crawl Data) từ Bonbanh.com

Bài viết này sẽ giải thích chi tiết cách viết một script Python để tự động lật trang và lấy thông tin xe ô tô từ trang web 4 bánh.

---

## 1. Import các thư viện cần thiết
Tại sao có cái mớ này? Đơn giản là chúng ta cần `requests` để gửi yêu cầu truy cập web, `BeautifulSoup` để đọc hiểu mã HTML, và `pandas` để xuất dữ liệu ra file CSV cho đẹp. `time` và `random` giúp tạo độ trễ để tránh bị khóa IP. Ngoài ra vốn vì cái HTML của cái web này nó khá là loạn nên việc dùng BS để xử lý thì nó sẽ ổn hơn và nó sắp xếp ngăn nắp lại cái HTML 

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
```

## 2. Hàm hỗ trợ lấy thông số chính xác
Bảng thông số kỹ thuật của trang web chứa rất nhiều hàng (`div` class `row`). Hàm `get_value_exact` này sẽ dò tìm chính xác tên nhãn (ví dụ: "Năm sản xuất", "Động cơ") và trả về giá trị tương ứng. Nếu không có, nó sẽ báo "N/A".

```python
def get_value_exact(soup, label_name):
    """Tìm thông số chi tiết của xe từ bảng kỹ thuật"""
    rows = soup.find_all('div', class_=['row', 'row_last'])
    for row in rows:
        label_div = row.find('div', class_='label')
        if label_div and label_name in label_div.text:
            input_div = row.find('div', class_=['txt_input', 'inputbox'])
            if input_div:
                return input_div.text.strip()
    return "N/A"
```

## 3. Hàm lấy chi tiết của một chiếc xe
Hàm `get_car_details` chịu trách nhiệm truy cập vào từng link xe cụ thể để lấy tất tần tật thông tin: Ngày đăng, Mô tả, Tên người bán, Địa chỉ,... 

Phần Tên xe và Giá được dính liền nhau trên web. Ở đây sử dụng Regex (`re.sub`) để chuẩn hóa các loại dấu gạch ngang khác nhau, giúp tách chuỗi một cách chính xác nhất.

`def get_car_details(session, url)`: Khai báo hàm. Nó nhận 2  cái là session (phiên duyệt web để giữ cookie, tránh bị chặn) và url (đường link của chiếc xe cần cào).

`try... except...`: Đây là "bảo hiểm" của code. Lỡ web sập, rớt mạng hay có lỗi bất ngờ, nó sẽ chạy vào except để in ra lỗi thay vì làm chết đứng cả chương trình.

`response = session.get(url, timeout=10)`: Mở đường link xe ra xem, nếu sau 10 giây web không phản hồi thì bỏ qua để khỏi treo máy.

`soup = BeautifulSoup(response.content, "html.parser")`: Đưa toàn bộ mã nguồn HTML của trang web cho thư viện BeautifulSoup nhào nặn thành một cái cây cấu trúc dễ tìm kiếm.
```python
def get_car_details(session, url):
    """Vào từng link lấy thông tin, bao gồm cả ngày đăng, mô tả, người bán"""
    try:
        response = session.get(url, timeout=10)
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.content, "html.parser")
```
## I.
Tại sao lại có cái hàm này? Vốn vì cái web này title nó là gộp cả giá và xe cùng 1 chỗ vậy nên phải dùng `title_tag = soup.find("h1"):` Tìm cái tiêu đề to nhất trang (thường chứa cả tên xe và giá). Thì sau đó dùng ``clean_title = re.sub(r'[-–—−]', '-', full_title)`` vì thường Các chủ xe gõ tiêu đề dùng đủ loại dấu gạch ngang (ngắn, dài, gạch nối...). Lệnh này gom tất cả các thể loại gạch ngang đó về đúng một chuẩn là dấu `-`

Lúc này thì sẽ dễ để tách ra hơn vì trang 4banh.com có hơi đặc biệt chút. ``parts = clean_title.rsplit("-", 1)``: Lệnh rsplit cắt chuỗi từ phải sang trái, cắt đúng 1 lần tại dấu gạch ngang cuối cùng. Cắt ở số 1 vì sẽ chia làm 2 phần, 0 là tên xe còn 1 là giá cả sau đó cứ strip
```python
        #Tên xe và Giá 
        title_tag = soup.find("h1")
        full_title = title_tag.text.strip() if title_tag else "N/A"
        clean_title = re.sub(r'[-–—−]', '-', full_title) 
        
        ten_xe = clean_title
        gia = "N/A"
        if "-" in clean_title:
            parts = clean_title.rsplit("-", 1) 
            ten_xe = parts[0].strip()
            gia = parts[1].strip()
```

`notes_div = soup.find('div', class_='notes')` Tìm cái khối có class là notes (khối chứa ngày đăng và số lượt xem).

`ngay_dang = notes_text.split('.')[0]...:` dòng text thường có dạng "Đăng ngày 12/10. Xem 50 lượt". Lệnh này sẽ chặt đứt tại dấu chấm . để vứt phần "Xem 50 lượt" đi, sau đó xóa nốt chữ "Đăng ngày" để chắt lọc lấy đúng cái ngày (VD: 12/10).

## II. lấy ngày đăng
```python
        #ngày đăng
        notes_div = soup.find('div', class_='notes')
        ngay_dang = "N/A"
        if notes_div:
            notes_text = notes_div.text.strip()
            if "Đăng ngày" in notes_text:
                ngay_dang = notes_text.split('.')[0].replace("Đăng ngày", "").strip()
```

## III. cách lấy tên và tách dòng đưa vào CPV
Thì dưới này có 1 cái hàm khá là lạ nó sẽ là `des_div.get_text(separator=" | ", strip=True)` thì tại sao có hàm này thì đơn giản nó sẽ là cái mà thay thế các xuống dòng là `<br>`, `\n` trong file inspect của trang để tự động thay bằng dấu `|` để đỡ cho việc nhảy dòng

tới đoạn `cname_tag = soup.find(class_='cname')` thì nó sẽ là lấy bất chấp những cái class bất chấp thẻ `<a>` hay `span` vì cứ có trong class `cname` là đưa về cpv hết

```python
        # Mô tả
        des_div = soup.find('div', class_='des_txt')
        mo_ta = des_div.get_text(separator=" | ", strip=True) if des_div else "N/A"
        
        # Tên người bán 
        cname_tag = soup.find(class_='cname')
        ten_nguoi_ban = cname_tag.text.strip() if cname_tag else "N/A"
```

## IV. Địa chỉ

Cái này thì có hơi phức tạp 1 tý tại vì phần địa chỉ nó sẽ lẫn với cả phần email và website sau khi check mã nguồn của nó. Thì đơn giản là gom tất tần tật chữ trong khối liên hệ lại thành một hàng ngang dài thòng lòng. sau đó thì cái dưới `match = re.search` thì dùng cái này vì sao  thì để dùng thuạt toán regrex để truy tìm dấu vết. Lệnh này dịch ra tiếng người là: "Hãy tìm chữ 'Địa chỉ:', sau đó lấy toàn bộ nội dung đằng sau nó, nhưng phải dừng lại ngay lập tức khi thấy chữ 'Website', chữ 'Email' hoặc hết dòng

```python        
        # ĐỊA CHỈ 
        dia_chi = "N/A"
        contact_div = soup.find('div', class_='contact-txt')
        if contact_div:
            raw_text = contact_div.get_text(separator=" ", strip=True)
            match = re.search(r'Địa chỉ\s*:\s*(.*?)(?:Website|Email|$)', raw_text, re.IGNORECASE)
            if match:
                dia_chi = match.group(1).strip()
                if not dia_chi: 
                    dia_chi = "N/A"
```
## V. Gom dữ liệu lại
```python
        # 6. Gom toàn bộ vào Dictionary
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
        print(f"Lỗi khi truy cập {url}: {e}")
        return None
```

## 3. Hàm Main: Quét toàn bộ và lưu dữ liệu
Nó hoạt động theo 2 bước:
- **Bước 1:** Tự động lật qua từng trang để gom tất cả các link URL xe có trên web (ở đây là mục xe điện).
- **Bước 2:** Chạy vòng lặp qua danh sách link vừa gom được, gọi hàm `get_car_details` để lấy thông số kỹ thuật, và cuối cùng dùng `pandas` lưu toàn bộ thành file `bonbanh.com.csv`.

* dùng headers cái này đơn giản là chỉ báo cho vên web biết là tôi là người này như này và tôi muốn access vào web bonbanh nhờ mã accept với cái referer để vào web. sau đó là sẽ dùng lệnh `request.session` để vào
```python
def crawl_bonbanh_all_pages_final():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "[https://bonbanh.com/](https://bonbanh.com/)"
    }

    session = requests.Session()
    session.headers.update(headers)

```
## I. access tới page

Đoạn này thì đơn giản hơn là chỉ lật từng trang, lấy tập hợp bằng mở 1 tập hợp rỗng tên `all_links` sau đó tọa thành 1 set bắt đầu từ page 1. Lúc này thì sẽ quét qua từng page 
```python
    all_links = []
    seen_links = set()
    page = 1
    
    print("Step 1: Tự động lật trang để gom link xe toàn web...")
    while True:
        if page == 1:
            url = "[https://bonbanh.com/oto-xe-dien](https://bonbanh.com/oto-xe-dien)"
        else:
            url = f"[https://bonbanh.com/oto-xe-dien/page](https://bonbanh.com/oto-xe-dien/page),{page}"
            
        print(f"Đang quét {url} ...")
```

Nhưng sẽ có 1 vài thứ cần phải chỉnh đó là nếu lỡ mà có lỗi mạng hay là gì vd như timeout hoặc miễn không phỉa lỗi 200 thì code sẽ về lại web trước và thử lại sau 3 giây đoạn `iff responese.status_code == 200` tới `success = true` là sẽ `break`. còn nếu không sẽ thử kết nối lại 
```python        
        retries = 3
        success = False
        while retries > 0:
            try:
                response = session.get(url, timeout=10)
                if response.status_code == 200:
                    success = True
                    break
                else:
                    print(f" Web phản hồi mã {response.status_code}. Thử lại sau 3s...")
                    time.sleep(3)
                    retries -= 1
            except Exception as e:
                print(f" Lỗi kết nối: {e}. Thử lại...")
                time.sleep(3)
                retries -= 1
                
        if not success:
            break
``` 

Sau khi xử lý mấy cái mớ bòng bong ở trên thì lúc này cái `beautifulsoup` nó sẽ sắp xếp lại cái mớ bòng bong ở trên thành 1 cấu trúc rành mạch để mà bóc tách từng lớp. Nhưng nó sẽ có 1 vấn đề là nếu không có `break` thì sẽ quay đúng 1 trang 1 cách vô tận. Nên mới xử lý là 
* `car_items = soup.find_all("li", class_="car-item")` Đoạn này là ném hết mấy cái ấy vào thẻ li chứa `class="car-item"`còn đoạn break là dùng khi mà để nó thoát ra khỏi cái link xe ấy, 
* lúc này máy sẽ xác nhận là tạo 1 tập hợp bắt dầu bằng `new_links_found = 0` sau đấy sẽ lôi từng cái link ra kiểm tra cho `for item in car_items:` và gộp chung cái full link để cộng xem đã tìm được bao nhiêu link
```python               
        soup = BeautifulSoup(response.content, "html.parser")
        car_items = soup.find_all("li", class_="car-item")
        
        if len(car_items) == 0:
            break
            
        new_links_found = 0
        for item in car_items:
            link_tag = item.find("a")
            if link_tag and 'href' in link_tag.attrs:
                full_link = "[https://bonbanh.com/](https://bonbanh.com/)" + link_tag['href']
```

Chỗ này thì sẽ tránh lặp lại những cái link đã được tìm thấy ở đấy bằng cách tạo 1 tập hợp là `seen_links` để xác nhận đã từng nhìn qua cái link ấy rồi thì sẽ không + 1 vào link mới tránh vào vòng lặp vô hạn khi mà code lấy link cũ.
```python               

                if full_link not in seen_links:
                    seen_links.add(full_link)
                    all_links.append(full_link)
                    new_links_found += 1
                    
        if new_links_found == 0:
            break
                
        page += 1
        time.sleep(random.uniform(1.0, 2.0)) 
```

Đoạn này đơn giảnn là gom link lại 1 thể thôi
```python
    print(f"\n=> TỔNG CỘNG: Gom được {len(all_links)} link xe điện.")
    print("\nStep 2: Bắt đầu lấy thông số...")
```

II. Khởi tạo
cái này khởi tạo 1 cái vỏ rỗng để đưa dữ liệu xe vào và sau đó theo `i` bắt đầu từ 1 thì nó sẽ xuất hiện tiến độ là đang cào được bao nhiêu xe. sau đó cứ theo `car_detail = ...` mà mò dữ liệu và nếu không bị lỗi sẽ đưa vào tập hợp ``all_cars_data`` sau đó ép nó dừng ngẫu nhiên từ 1-2 giây
```python
    all_cars_data = []
    
    for i, link in enumerate(all_links):
        print(f"Đang cào dữ liệu ({i+1}/{len(all_links)}): {link}")
        
        car_detail = get_car_details(session, link)
        if car_detail:
            all_cars_data.append(car_detail)
        
        time.sleep(random.uniform(1.0, 2.0))
```

III. đóng gói dữ liệu

phần code cứng đã qua thì chỉ có đóng gói lại dữ liệu, tạo cột các kiểu thôi và xuất ra file
```python
    if all_cars_data:
        df = pd.DataFrame(all_cars_data)
        cols = ["Ngày đăng", "Tên xe", "Giá", "Tên người bán", "Địa chỉ", "Năm sản xuất", "Tình trạng", "Số Km đã đi", "Xuất xứ", "Kiểu dáng", "Hộp số", "Động cơ", "Màu ngoại thất", "Màu nội thất", "Số chỗ ngồi", "Số cửa", "Dẫn động", "Mô tả", "Link"]
        df = df[cols]
        
        df.to_csv("bonbanh.com.csv", index=False, encoding="utf-8-sig")
        print(f"\n XONG! Đã quét sạch {len(all_cars_data)} xe.")

if __name__ == "__main__":
    crawl_bonbanh_all_pages_final()
```