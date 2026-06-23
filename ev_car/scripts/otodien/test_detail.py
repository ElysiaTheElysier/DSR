import re
import json
import requests

REQ_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
}

def get_detail(url):
    res = requests.get(url, headers=REQ_HEADERS)
    html = res.text
    
    matches = re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"]\)', html)
    for m in matches:
        try:
            text = m.encode('utf-8').decode('unicode_escape')
            if '"items":{' in text:
                # Trích xuất đoạn JSON chứa items (là object)
                # Next.js thường bọc items trong 1 object json response {"status":"I'm ok!","items":{...}}
                match = re.search(r'"items":(\{.*?\})(?:,"others"|,"similar"|\})', text)
                if match:
                    json_str = match.group(1)
                    item = json.loads(json_str)
                    print(f"Thành công! ID: {item.get('id')}, Tên: {item.get('title')}, Pin: {item.get('battery')}")
                    return item
        except Exception as e:
            pass
    print("Không tìm thấy JSON.")
    return None

if __name__ == '__main__':
    get_detail('https://otodien.vn/oto-vinfast-vf7-nam-2026/100001974')
