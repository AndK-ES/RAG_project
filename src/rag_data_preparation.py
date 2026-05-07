import os
import json
from typing import List, Dict, Any
import tiktoken

CHUNK_SIZE = 800
CHUNK_OVERLAP = 80

enc = tiktoken.get_encoding('cl100k_base')

def get_all_json_files(root_dir: str) -> List[str]:
    json_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith('.json'):
                json_files.append(os.path.join(dirpath, f))
    return json_files

def extract_text_from_department(data: Dict[str, Any]) -> str:
    return data.get('name', '')

def extract_text_from_object(data: Dict[str, Any]) -> str:
    fields = [
        'artist', 'classification', 'continent', 'country', 'creditline', 'dated',
        'department', 'description', 'dimension', 'medium', 'object_name', 'portfolio',
        'provenance', 'signed', 'style', 'text', 'title'
    ]
    return '\n'.join(str(data.get(f, '')) for f in fields if data.get(f))

def extract_text_from_exhibition(data: Dict[str, Any]) -> str:
    parts = [
        data.get('exhibition_title', ''),
        data.get('exhibition_description', ''),
        data.get('public_info', ''),
        data.get('display_date', '')
    ]
    venues = data.get('venues', [])
    for v in venues:
        if isinstance(v, dict):
            parts.append(v.get('venue', ''))
            parts.append(v.get('display_date', ''))
    return '\n'.join(str(p) for p in parts if p)

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = enc.decode(tokens[start:end])
        chunks.append(chunk)
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks

def main():
    root = 'collection-main'
    json_files = get_all_json_files(root)
    all_chunks = []
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
        data_items = data if isinstance(data, list) else [data]
        for item in data_items:
            if not isinstance(item, dict):
                continue
            if '/departments/' in file_path.replace('\\', '/').lower():
                text = extract_text_from_department(item)
            elif '/objects/' in file_path.replace('\\', '/').lower():
                text = extract_text_from_object(item)
            elif '/exhibitions/' in file_path.replace('\\', '/').lower():
                text = extract_text_from_exhibition(item)
            else:
                continue
            if text.strip():
                chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        'source_file': file_path,
                        'chunk_id': i,
                        'text': chunk
                    })
    with open('data_chunks.json', 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved chunks: {len(all_chunks)}")

if __name__ == '__main__':
    main()
