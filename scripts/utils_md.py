# scripts/utils_md.py
import re

def split_by_headers(md_text):
    pattern = r"(^#{2,3} .+$)"
    parts = re.split(pattern, md_text, flags=re.MULTILINE)

    chunks = []
    current_header = None
    current_body = []

    for part in parts:
        if part.startswith("##"):
            if current_header:
                chunks.append((current_header, "\n".join(current_body)))
            current_header = part.strip()
            current_body = []
        else:
            current_body.append(part.strip())

    if current_header:
        chunks.append((current_header, "\n".join(current_body)))

    return chunks
