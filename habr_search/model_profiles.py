"""храним параметры bge-m3, общие для индексации и поиска."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from habr_search.paths import DATA_DIR, MODELS_DIR


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model_id: str
    revision: str
    vector_size: int
    deployment: str
    bos_token: str
    bos_id: int
    eos_token: str
    eos_id: int
    query_max_tokens: int = 1024  # предел поискового запроса для деплоймента bge-m3

    @property
    def tokenizer_file(self) -> Path:
        """кеш разных моделей и ревизий никогда не пересекается."""
        return MODELS_DIR / self.model_id / self.revision / 'tokenizer.json'

    @property
    def dictionary_file(self) -> Path:
        """используем словарь, построенный из коллекции bge-m3."""
        return DATA_DIR / 'spelling-bge-m3.sqlite3'


BGE_M3: ModelProfile = ModelProfile(
    'bge-m3', 'BAAI/bge-m3', '5617a9f61b028005a4858fdac845db406aefb181',
    1024, 'bge-m3', '<s>', 0, '</s>', 2,
)
# основной профиль выбран по независимой проверке качества и скорости
DEFAULT_PROFILE: ModelProfile = BGE_M3

PROFILE_NAMES: tuple[str, ...] = ('bge-m3',)


def get_profile(name: str) -> ModelProfile:
    """не пытаемся угадать модель по имени деплоймента или длине ответа api."""
    if name == 'bge-m3':
        return BGE_M3
    raise ValueError(f'неизвестный профиль модели: {name}')


def deployment_name(profile: ModelProfile) -> str:
    """при необходимости имя деплоймента можно явно переопределить окружением."""
    return os.environ.get('MWS_EMBEDDINGS_MODEL', profile.deployment)
