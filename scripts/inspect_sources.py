"""Audit fetched source content and recover official download links."""
from html.parser import HTMLParser
from pathlib import Path
import json
import zipfile


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(v for k,v in attrs if k == "href")


for key in ("gqa", "evict", "reality_check", "coev"):
    p = Path(f"outputs/sources/{key}.source")
    parser = Links(); parser.feed(p.read_text(encoding="utf-8"))
    print(key, [u for u in parser.links if any(s in u for s in (".pdf", ".zip", "github", "license"))])

p = Path("outputs/refcocog-archive-attempt.bin")
if zipfile.is_zipfile(p):
    with zipfile.ZipFile(p) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            path = Path(name)
            assert not path.is_absolute() and ".." not in path.parts
        archive.extractall("datasets")
    print("RefCOCOg archive CRC validated, extracted locally")
