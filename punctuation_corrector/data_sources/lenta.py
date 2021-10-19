import pandas as pd
from pathlib import Path
from typing import List


def read_lenta(path: Path) -> List[str]:
    df = pd.read_csv(path)
    return list(df['text'].astype('str'))
