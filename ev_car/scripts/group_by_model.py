import pandas as pd
import re

def clean_model_name(name):
    name = str(name).strip()
    
    # Common prefixes to remove
    name = re.sub(r'^(Xe máy điện|Xe đạp điện|Bán xe máy điện|Xe ô tô điện|Xe điện|Xe|Bán|Mua|Oto)\s+', '', name, flags=re.IGNORECASE).strip()
    
    # Colors to remove from end
    colors = ['Trắng', 'Đen', 'Đỏ', 'Xanh', 'Xám', 'Bạc', 'Hồng', 'Vàng', 'Cam']
    for color in colors:
        if name.lower().endswith(color.lower()):
            name = name[:-len(color)].strip()
            
    # Remove years
    name = re.sub(r'\b20[12]\d\b', '', name)
            
    # Normalize VF names (e.g., VF 5 -> VF5, VFe34 -> VF e34)
    name = re.sub(r'VF\s+(\d)', r'VF\1', name, flags=re.IGNORECASE)
    
    # Remove noise like "cũ", "lướt", "siêu lướt"
    name = re.sub(r'\b(cũ|lướt|siêu lướt|chính chủ|bản|bán)\b', '', name, flags=re.IGNORECASE)
    
    # Extract base brand + model
    parts = [p for p in name.split() if p.strip() and p.isalnum()]
    if len(parts) >= 2:
        if parts[0].lower() == 'vinfast':
            # keep up to 2 parts for cars: VinFast VF5
            if len(parts) >= 2 and parts[1].lower().startswith('vf'):
                return f"VinFast {parts[1].upper()}"
            elif len(parts) >= 3 and parts[1].lower() in ['feliz', 'klara', 'vento', 'evo200']:
                return " ".join(parts[:3]).title() # e.g. Vinfast Feliz S
            else:
                return " ".join(parts[:2]).title()
        elif parts[0].lower() == 'vf':
            return f"VinFast {parts[0].upper()}{parts[1]}" if len(parts[1]) == 1 else f"VinFast {parts[0].upper()} {parts[1].title()}"
        else:
            return " ".join(parts[:2]).title()
    return name.title()

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    df_oto = pd.read_csv('data/interim/ev_cleaned_oto.csv')
    df_bike = pd.read_csv('data/interim/ev_cleaned_bike.csv')

    df = pd.concat([df_oto, df_bike], ignore_index=True)
    df['Model_Clean'] = df['Tên xe'].apply(clean_model_name)

    grouped = df.groupby('Model_Clean').agg(
        Count=('Giá_VND', 'count'),
        Avg_Price_VND=('Giá_VND', 'mean'),
        Avg_Mileage_Km=('Số Km đã đi', 'mean')
    ).reset_index()

    grouped = grouped[grouped['Count'] > 5] # Only show models with at least 5 listings
    grouped = grouped.sort_values(by='Count', ascending=False)
    
    grouped['Avg_Price_VND'] = grouped['Avg_Price_VND'].apply(lambda x: f"{x:,.0f} VNĐ" if pd.notna(x) else "N/A")
    grouped['Avg_Mileage_Km'] = grouped['Avg_Mileage_Km'].apply(lambda x: f"{x:,.0f} Km" if pd.notna(x) else "N/A")
    
    print(grouped.to_markdown(index=False))
