import csv
import os

infile = 'C:/Ki_5/ev_car/data/interim/ev_extracted_qwen2.5_3b.csv'
outfile = infile + '.fixed'

with open(infile, 'r', encoding='utf-8-sig') as fin, open(outfile, 'w', encoding='utf-8-sig', newline='') as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    
    # Read the header
    header = next(reader)
    if len(header) == 10:
        header.append('vehicle_type')
    writer.writerow(header)
    
    for row in reader:
        if len(row) == 10:
            row.append('') # pad vehicle_type
            writer.writerow(row)
        elif len(row) == 11:
            writer.writerow(row)
        else:
            # Maybe some corrupt rows, pad or truncate
            if len(row) < 11:
                row.extend([''] * (11 - len(row)))
            else:
                row = row[:11]
            writer.writerow(row)

os.replace(outfile, infile)
print("CSV fixed successfully.")
