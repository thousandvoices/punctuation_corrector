from pathlib import Path
from typing import List
from lxml import etree


def read_fb2_collection(path: Path) -> List[str]:
    files = [f for f in path.iterdir() if str(f).endswith('fb2')]
    result = []

    for filename in files:
        with open(filename) as f:
            try:
                tree = etree.parse(f)
                if len(tree.xpath("//*[name()='publisher']")) > 0:
                    result.extend(tree.xpath("//*[name()='p']/text()"))
            except Exception:
                pass

    return result
