import re
import csv
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime

# -------- CONFIG --------
DOMAIN = "https://xevinfastluot.com"
LIST_URL = DOMAIN + "/bang-gia-xe-vinfast-luot/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "vi,en;q=0.9",
}

CSV_FIELDS = [
    "Ngày đăng","Tên xe","Giá","Tên người bán","Địa chỉ","Năm sản xuất",
    "Tình trạng","Số Km đã đi","Xuất xứ","Kiểu dáng","Hộp số",
    "Động cơ","Màu ngoại thất","Màu nội thất","Số chỗ ngồi",
    "Số cửa","Dẫn động","Mô tả","Link"
]

LABEL_MAP = {
    "Xuất xứ": "Xuất xứ",
    "Tình trạng": "Tình trạng",
    "Dòng xe": "Kiểu dáng",
    "Số km đã đi": "Số Km đã đi",
    "Màu ngoại thất": "Màu ngoại thất",
    "Màu nội thất": "Màu nội thất",
    "Số cửa": "Số cửa",
    "Số chỗ ngồi": "Số chỗ ngồi",
    "Hộp số": "Hộp số",
    "Động cơ": "Động cơ",
    "Dẫn động": "Dẫn động"
}

# -------- HELPERS --------

def clean(txt):
    return re.sub(r"\s+", " ", txt.strip()) if txt else ""

def get_soup(url, session):
    r = session.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")

# -------- SITEMAP --------

def try_fetch_sitemap_urls(base_url, session):
    candidates = [
        "/wp-sitemap.xml",
        "/sitemap.xml",
        "/sitemap_index.xml"
    ]
    sitemap_urls = []

    for path in candidates:
        try:
            url = urljoin(base_url, path)
            r = session.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200 and "<urlset" in r.text:
                sitemap_urls.append(url)
        except:
            pass

    # check robots.txt
    try:
        robots = session.get(base_url + "/robots.txt", headers=HEADERS, timeout=8).text
        for line in robots.splitlines():
            if line.lower().startswith("sitemap:"):
                sitemap_urls.append(line.split(":",1)[1].strip())
    except:
        pass

    return list(dict.fromkeys(sitemap_urls))

def parse_sitemap_xml(xml_text):
    mapping = {}
    try:
        root = ET.fromstring(xml_text.encode("utf-8"))
    except ET.ParseError:
        return mapping

    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    # nested sitemapindex
    for smap in root.findall("sm:sitemap", ns):
        loc = smap.find("sm:loc", ns)
        if loc is not None:
            try:
                r = requests.get(loc.text, headers=HEADERS, timeout=10)
                mapping.update(parse_sitemap_xml(r.text))
            except:
                pass

    for url in root.findall("sm:url", ns):
        u = url.find("sm:loc", ns)
        lm = url.find("sm:lastmod", ns)
        if u is not None:
            mapping[u.text] = lm.text if (lm is not None and lm.text) else ""

    return mapping

def build_sitemap_map(session):
    sitemap_map = {}
    urls = try_fetch_sitemap_urls(DOMAIN, session)
    for su in urls:
        try:
            resp = session.get(su, headers=HEADERS, timeout=15)
            sitemap_map.update(parse_sitemap_xml(resp.text))
        except:
            pass
    return sitemap_map

# -------- REST API CPT --------

def get_slug(url):
    return urlparse(url).path.strip("/").split("/")[-1]

def try_rest_api_cpt(slug, session):
    """
    Try to lookup slug from possible REST API CPT endpoints
    (common WP naming patterns).
    """
    candidates = [
        "san-pham",   # try direct post_type if exposed
        "products"    # common ecommerce naming
    ]
    for base in candidates:
        api_url = f"{DOMAIN}/wp-json/wp/v2/{base}?slug={slug}"
        try:
            r = session.get(api_url, headers=HEADERS, timeout=10)
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                # expect "date" field
                d = data[0].get("date", data[0].get("post_date", ""))
                if d:
                    try:
                        dt = datetime.fromisoformat(d.replace("Z","+00:00"))
                        return dt.strftime("%d/%m/%Y")
                    except:
                        return ""
        except:
            pass
    return ""

# -------- SCRAPE DETAIL --------

def extract_detail_links(list_url, session):
    soup = get_soup(list_url, session)
    links = set()
    for a in soup.select("a[href*='/san-pham/']"):
        links.add(urljoin(list_url, a["href"]).split("#")[0])
    return sorted(links)

def parse_detail(url, session, sitemap_map):
    soup = get_soup(url, session)
    row = {c:"" for c in CSV_FIELDS}
    row["Link"] = url

    # 1) ngày đăng: API CPT
    slug = get_slug(url)
    api_date = try_rest_api_cpt(slug, session)
    # 2) fallback sitemap lastmod
    sitemap_date = ""
    if not api_date:
        lm = sitemap_map.get(url, "")
        if lm:
            try:
                # try to parse ISO-like
                dt = datetime.fromisoformat(lm.replace("Z","+00:00"))
                sitemap_date = dt.strftime("%d/%m/%Y")
            except:
                # fallback simple iso
                m = re.search(r"(\d{4})-(\d{2})-(\d{2})", lm)
                if m:
                    sitemap_date = f"{m.group(3)}/{m.group(2)}/{m.group(1)}"
    row["Ngày đăng"] = api_date or sitemap_date

    # 3) Tên xe / giá
    title_el = soup.select_one("h2.title_singleproduct") or soup.select_one("h1")
    txt = clean(title_el.text) if title_el else ""
    if " - " in txt:
        name,price = txt.split(" - ",1)
    else:
        name,price = txt,""
    row["Tên xe"], row["Giá"] = clean(name), clean(price)

    # năm sản xuất
    m = re.search(r"\b(19\d{2}|20\d{2})\b", row["Tên xe"])
    row["Năm sản xuất"] = m.group(0) if m else ""

    # seller & address
    seller_marker = soup.find(string=re.compile(r"Liên hệ người bán", re.I))
    if seller_marker:
        parent = seller_marker.find_parent()
        h = parent.find_next(["h1","h2","h3","h4","h5"])
        row["Tên người bán"] = clean(h.text) if h else ""
        addr_node = parent.find_next(string=re.compile(r"Địa chỉ", re.I))
        row["Địa chỉ"] = clean(addr_node.split(":",1)[-1]) if addr_node else ""

    # mô tả
    desc_marker = soup.find(string=re.compile(r"Thông tin mô tả", re.I))
    if desc_marker:
        p = desc_marker.find_parent().find_next("p")
        row["Mô tả"] = clean(p.text) if p else ""

    # bảng thông số
    table_kv = {}
    for tr in soup.select("table tr"):
        tds = tr.find_all(["td","th"])
        if len(tds)>=2:
            table_kv[clean(tds[0].text).rstrip(":")] = clean(tds[1].text)
    for label, col in LABEL_MAP.items():
        if label in table_kv:
            row[col] = table_kv[label]

    return row

# -------- MAIN --------

def main():
    with requests.Session() as session:
        sitemap_map = build_sitemap_map(session)
        print(f"Sitemap entries: {len(sitemap_map)}")

        detail_links = extract_detail_links(LIST_URL, session)
        print(f"Found {len(detail_links)} detail links to crawl")

        with open("xevinfastluot_full.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for url in tqdm(detail_links, desc="Scraping"):
                try:
                    row = parse_detail(url, session, sitemap_map)
                    writer.writerow(row)
                except Exception as e:
                    print(f"[ERR] {url}: {e}")

    print("Done!")

if __name__ == "__main__":
    main()