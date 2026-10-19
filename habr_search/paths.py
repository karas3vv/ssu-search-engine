"""храним общие пути независимо от папки запуска и расположения модулей."""
from pathlib import Path

# корпус, словарь и токенизатор остаются в прежних папках в корне проекта
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / 'data'
MODELS_DIR: Path = PROJECT_ROOT / 'models'
