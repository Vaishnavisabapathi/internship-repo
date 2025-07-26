from services.segment_lines import segment_all_pages

result = segment_all_pages("iot1_016.pdf", offset=0, limit=5)
for line in result:
    print(line["line_id"], line["ocr_text"][:50])
