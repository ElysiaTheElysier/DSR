import re
import json

html = open('scratch/otodien_listings.html', encoding='utf-8').read()
matches = re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"]\)', html)
for m in matches:
    # Next.js escapes quotes in the push strings
    text = m.encode('utf-8').decode('unicode_escape')
    if '"items":[{' in text:
        print('Found items!')
        # Use regex to extract the JSON object containing "items"
        # Since it's a deeply nested JSON string inside the RSC payload, we can extract it by finding the outermost matching braces.
        # But for now, just finding the index of '"items":[{' is enough to prove it works.
        idx = text.find('"items":[{')
        print(text[idx:idx+200])
