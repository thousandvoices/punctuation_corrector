import urllib
import hashlib
from zipfile import ZipFile
from tempfile import NamedTemporaryFile
from contextlib import contextmanager
from pathlib import Path


PROTOCOL_SEPARATOR = '://'


@contextmanager
def download(source: str):
    with NamedTemporaryFile() as archive:
        urllib.request.urlretrieve(source, archive.name)
        yield archive.name


@contextmanager
def copy(source: str):
    yield source.split(PROTOCOL_SEPARATOR, 1)[-1]


PROTOCOL_HANDLERS = {
    'http': download,
    'https': download,
    'file': copy
}


class ModelCache:
    def __init__(self, cache_path: str) -> None:
        self._path = Path(cache_path)

    def cached_path(self, model_name: str) -> Path:
        protocol = model_name.split(PROTOCOL_SEPARATOR)[0]
        downloader = PROTOCOL_HANDLERS.get(protocol)

        if downloader is not None:
            self._path.mkdir(parents=True, exist_ok=True)
            cache_key = hashlib.md5(model_name.encode('utf-8')).hexdigest()
            cache_path = self._path / cache_key
            if cache_path.exists():
                return cache_path
            else:
                with downloader(model_name) as archive_path, ZipFile(archive_path) as archive:
                    archive.extractall(cache_path)

                return cache_path
        else:
            return Path(model_name)
