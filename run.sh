#!/usr/bin/env bash
# запускаем базу и страницу поиска одной командой
set -euo pipefail

# пути к окружению и данным считаем от папки проекта
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -x .venv/bin/python ]]; then
    python_bin="$PWD/.venv/bin/python"
else
    python_bin=python3
fi

# локальный режим запускаем без docker, ключа и проверки qdrant
previous_argument=''
for argument in "$@"; do
    if [[ "$argument" == "--engine=tantivy" || ( "$previous_argument" == "--engine" && "$argument" == "tantivy" ) ]]; then
        exec "$python_bin" -m habr_search.web.app "$@"
    fi
    previous_argument="$argument"
done

docker compose up -d qdrant

# ждём ответа базы, чтобы первый запуск не опередил загрузку qdrant
"$python_bin" - <<'PY'
import time
from urllib.error import URLError
from urllib.request import OpenerDirector, ProxyHandler, build_opener

client: OpenerDirector = build_opener(ProxyHandler({}))
for attempt in range(30):
    try:
        with client.open('http://127.0.0.1:6333/collections', timeout=1):
            break
    except (URLError, OSError):
        time.sleep(1)
else:
    raise SystemExit('qdrant не ответил; проверьте контейнер: docker compose logs qdrant')
PY

# передаём управление серверу, чтобы ctrl+c завершал приложение
exec "$python_bin" -m habr_search.web.app \
    --qdrant-url http://127.0.0.1:6333 \
    "$@"
