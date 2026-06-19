import csv
import os

infile = 'C:/Ki_5/ev_car/data/interim/ev_extracted_qwen2.5_3b.csv'
outfile = infile + '.tmp'
good_count = 0
bad_count = 0

with open(infile, 'r', encoding='utf-8-sig') as fin:
    reader = csv.reader(fin)
    header = next(reader)
    rows = [row for row in reader]

with open(outfile, 'w', encoding='utf-8-sig', newline='') as fout:
    writer = csv.writer(fout)
    writer.writerow(header)
    for row in rows:
        if len(row) == 10:
            writer.writerow(row)
            good_count += 1
        else:
            bad_count += 1

os.replace(outfile, infile)
print(f"Kept {good_count} rows, dropped {bad_count} bad rows.")
