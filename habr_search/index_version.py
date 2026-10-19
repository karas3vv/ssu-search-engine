"""связываем словарь и отчёты с версией содержимого коллекции qdrant."""
from uuid import uuid4

from qdrant_client import QdrantClient

UNVERSIONED: str = 'unversioned'
VERSION_KEY: str = 'habr_index_revision'


def index_revision(store: QdrantClient, name: str) -> str:
    """читаем общую версию; старые коллекции получают явную отметку без записи в индекс."""
    metadata: dict[str, object] = dict(store.get_collection(name).config.metadata or {})
    value = metadata.get(VERSION_KEY, UNVERSIONED)
    if not isinstance(value, str) or not value:
        raise ValueError('повреждена версия индекса в метаданных коллекции')
    return value


def invalidate_index(store: QdrantClient, name: str) -> str:
    """меняем версию до записи, чтобы сбой после неё тоже сделал прежний словарь устаревшим."""
    metadata: dict[str, object] = dict(store.get_collection(name).config.metadata or {})
    revision = uuid4().hex
    metadata[VERSION_KEY] = revision
    if not store.update_collection(name, metadata=metadata):
        raise RuntimeError('не удалось обновить версию индекса; запись фрагментов отменена')
    return revision
