"""проверяем доступ к удалённой модели и получаем эмбеддинги через api mws."""
# вход: строки текста; выход: числовые векторы удалённой модели
from __future__ import annotations

import argparse
import getpass
import math
import os
import sys
from contextlib import closing
from dataclasses import dataclass, field
from typing import cast
from urllib.parse import urlsplit

import requests

DEFAULT_URL: str = 'https://gpt.mwsapis.ru/projects/habr-parser/openai/v1/embeddings'
DEFAULT_MODEL: str = 'parser'  # имя деплоймента в mws, а не имя модели на hugging face


class EmbeddingAPIError(RuntimeError):
    """ошибка подключения или некорректный ответ сервера."""


# настройки удалённого деплоймента; ключ скрыт из строкового представления
@dataclass(frozen=True)
class EmbeddingConfig:
    api_key: str = field(repr=False)  # ключ не выводится при печати настроек
    url: str = DEFAULT_URL  # адрес метода получения векторов
    model: str = DEFAULT_MODEL  # имя деплоймента на сервере mws
    timeout_seconds: float = 120.0  # ожидание данных ответа; подключение ограничено отдельно

    # проверяем ключ, адрес сервиса, имя деплоймента и допустимое ожидание
    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError('api-ключ не указан')
        parsed = urlsplit(self.url)
        if (parsed.scheme != 'https' or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError('нужен https-адрес embeddings без пароля, параметров и фрагмента')
        if not self.model.strip():
            raise ValueError('имя деплоймента не указано')
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError('тайм-аут должен быть положительным конечным числом')

    # берём адрес и имя деплоймента из окружения, используя переданный ключ
    @classmethod
    def from_env(cls, api_key: str) -> EmbeddingConfig:
        return cls(
            api_key=api_key,
            url=os.environ.get('MWS_EMBEDDINGS_URL', DEFAULT_URL),
            model=os.environ.get('MWS_EMBEDDINGS_MODEL', DEFAULT_MODEL),
        )


# проверяем, что часть ответа api является объектом, и уточняем её тип
def json_object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EmbeddingAPIError(f'{name}: ожидался объект json')
    return cast(dict[str, object], value)


def parse_vectors(payload: object, expected_count: int) -> list[list[float]]:
    """проверяем ответ и восстанавливаем порядок векторов по полю index."""
    root = json_object(payload, 'ответ')
    value = root.get('data')
    if not isinstance(value, list):
        raise EmbeddingAPIError('в ответе нет списка data')
    items = cast(list[object], value)
    if len(items) != expected_count:
        raise EmbeddingAPIError('число векторов не совпадает с числом отправленных текстов')
    # собираем векторы по номерам из ответа; порядок списка data может отличаться
    vectors: dict[int, list[float]] = {}
    dimension: int | None = None
    for item in items:
        record = json_object(item, 'элемент data')
        index = record.get('index')
        if type(index) is not int or not 0 <= index < expected_count or index in vectors:
            raise EmbeddingAPIError('неверный или повторяющийся index в ответе')
        raw_vector = record.get('embedding')
        if not isinstance(raw_vector, list) or not raw_vector:
            raise EmbeddingAPIError('embedding должен быть непустым списком чисел')
        vector: list[float] = []
        for coordinate in cast(list[object], raw_vector):
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                raise EmbeddingAPIError('embedding содержит нечисловое значение')
            try:
                number = float(coordinate)
            except OverflowError as exc:
                raise EmbeddingAPIError('embedding содержит слишком большое число') from exc
            if not math.isfinite(number):
                raise EmbeddingAPIError('embedding содержит nan или бесконечность')
            vector.append(number)
        if not any(vector):
            raise EmbeddingAPIError('сервер вернул нулевой вектор')
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise EmbeddingAPIError('размерности векторов в ответе различаются')
        vectors[index] = vector
    return [vectors[index] for index in range(expected_count)]


# клиент переводит переданные строки в векторы через api mws
class EmbeddingsClient:
    # создаём сессию для повторного использования соединений
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config: EmbeddingConfig = config
        self.session: requests.Session = requests.Session()

    # освобождаем соединения после завершения работы с клиентом
    def close(self) -> None:
        self.session.close()

    # отправляем пакет строк и возвращаем проверенные векторы в том же порядке
    def embed(self, texts: list[str]) -> list[list[float]]:
        # формат входных строк задаётся вызывающим кодом: клиент не добавляет префиксы e5
        if not texts or any(not text.strip() for text in texts):
            raise ValueError('нужен непустой список непустых текстов')
        try:
            response = self.session.post(
                self.config.url,
                headers={
                    'Authorization': f'Bearer {self.config.api_key}',
                    'Content-Type': 'application/json',
                },
                json={'model': self.config.model, 'input': texts},
                timeout=(10, self.config.timeout_seconds),
                allow_redirects=False,
            )
        except requests.ConnectTimeout:
            raise EmbeddingAPIError(
                'тайм-аут подключения к MWS (10 с); запрос автоматически не повторяется'
            ) from None
        except requests.ReadTimeout:
            raise EmbeddingAPIError(
                f'тайм-аут ожидания данных ответа MWS ({self.config.timeout_seconds:g} с); '
                'запрос автоматически не повторяется'
            ) from None
        except requests.Timeout:
            raise EmbeddingAPIError(
                f'истёк тайм-аут MWS (подключение: 10 с; ожидание ответа: {self.config.timeout_seconds:g} с); '
                'запрос автоматически не повторяется'
            ) from None
        except requests.RequestException:
            # исходное исключение и тело ответа могут содержать лишние служебные данные
            raise EmbeddingAPIError('не удалось подключиться к api; проверьте сеть и адрес сервера') from None
        with response:
            status = response.status_code
            if not 200 <= status < 300:
                explanations: dict[int, str] = {
                    400: 'проверьте формат запроса и ограничения модели',
                    401: 'проверьте api-ключ сервисного аккаунта',
                    403: 'проверьте доступ к проекту и роль gpt.inferencer',
                    404: 'проверьте адрес проекта и имя деплоймента',
                    429: 'превышена квота или частота запросов; повторите позже',
                }
                explanation = explanations.get(status, 'ошибка сервера или неожиданный ответ')
                raise EmbeddingAPIError(f'http {status}: {explanation}')
            try:
                payload: object = response.json()
            except ValueError:
                raise EmbeddingAPIError('сервер вернул ответ не в формате json') from None
            # проверка формата и чисел завершается до передачи векторов вызывающему коду
            return parse_vectors(payload, len(texts))


# параметры отдельной проверки подключения к модели
class CliArgs(argparse.Namespace):
    text: str


def read_api_key() -> str:
    """получаем ключ из окружения или скрытого ввода для всех команд проекта."""
    api_key = os.environ.get('MWS_API_KEY', '').strip()
    if api_key:
        return api_key
    if not sys.stdin.isatty():
        raise ValueError('задайте MWS_API_KEY или запустите команду в интерактивном терминале')
    try:
        api_key = getpass.getpass('api-ключ сервисного аккаунта mws: ').strip()
    except (EOFError, KeyboardInterrupt):
        raise ValueError('ввод ключа отменён') from None
    if not api_key:
        raise ValueError('api-ключ не указан')
    return api_key


# проверяем один текст и показываем размерность полученного вектора
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', default='Привет, как дела?', help='один текст для проверки api')
    args: CliArgs = parser.parse_args(namespace=CliArgs())
    if not args.text.strip():
        parser.error('текст не должен быть пустым')
    try:
        api_key = read_api_key()
    except ValueError as exc:
        parser.exit(1, f'{exc}\n')
    try:
        config = EmbeddingConfig.from_env(api_key)
        with closing(EmbeddingsClient(config)) as client:
            vector = client.embed([args.text])[0]
    except (ValueError, EmbeddingAPIError) as exc:
        parser.exit(1, f'не удалось получить эмбеддинг: {exc}\n')
    print('подключение к api работает')
    print(f'деплоймент: {config.model}')
    print(f'размерность вектора: {len(vector)}')
    print(f'первые 5 координат: {vector[:5]}')
    print('корпус статей не читался и не изменялся')
    return 0


# при прямом запуске выполняем команду; при импорте этот блок пропускается
if __name__ == '__main__':
    raise SystemExit(main())
