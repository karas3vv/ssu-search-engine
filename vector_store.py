"""единые настройки подключения к серверному или встроенному qdrant."""
# общие настройки соединения используются индексатором, поиском и оценкой
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from qdrant_client import QdrantClient

DEFAULT_QDRANT_PATH: Path = Path(__file__).resolve().parent / 'data' / 'qdrant'


# выбираем один режим хранения: адрес сервера либо локальную папку
@dataclass(frozen=True)
class QdrantConfig:
    url: str | None = None  # адрес отдельного сервера
    path: Path | None = None  # папка встроенного режима
    api_key: str | None = field(default=None, repr=False)  # нужен только для защищённого сервера

    # исключаем противоречивые настройки и передачу ключа по открытому удалённому http
    def __post_init__(self) -> None:
        if (self.url is None) == (self.path is None):
            raise ValueError('задайте либо адрес qdrant, либо локальную папку')
        if self.url is not None:
            parsed = urlsplit(self.url)
            if (parsed.scheme not in ('http', 'https') or not parsed.hostname
                    or parsed.username or parsed.password or parsed.query or parsed.fragment):
                raise ValueError('нужен http(s)-адрес qdrant без логина, пароля и параметров')
            if (self.api_key and parsed.scheme != 'https'
                    and parsed.hostname not in ('localhost', '127.0.0.1', '::1')):
                raise ValueError('для передачи ключа удалённому qdrant используйте https')


# сначала учитываем явные аргументы, затем окружение и локальное значение по умолчанию
def resolve_qdrant_config(url: str | None, path: Path | None) -> QdrantConfig:
    # явный аргумент командной строки важнее переменной окружения
    if url is not None and path is not None:
        raise ValueError('нельзя одновременно задавать qdrant-url и qdrant-path')
    if path is not None:
        return QdrantConfig(path=path)
    endpoint = url if url is not None else os.environ.get('QDRANT_URL')
    if endpoint is not None:
        return QdrantConfig(url=endpoint, api_key=os.environ.get('QDRANT_API_KEY') or None)
    return QdrantConfig(path=DEFAULT_QDRANT_PATH)


# создаём клиент выбранного режима; закрывать его должен вызывающий код
def connect_store(config: QdrantConfig) -> QdrantClient:
    if config.url is not None:
        return QdrantClient(url=config.url, api_key=config.api_key, timeout=60)
    return QdrantClient(path=str(config.path))
