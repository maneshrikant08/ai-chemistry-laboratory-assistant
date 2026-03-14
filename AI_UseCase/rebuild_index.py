from pathlib import Path

from config import config
from models.embeddings import get_embeddings
from utils.vector_store import build_index_from_data


def main():
    try:
        data_dir = config.DATA_DIR
        index_dir = config.INDEX_DIR
        if not data_dir.exists():
            raise RuntimeError(f"Data directory not found: {data_dir}")

        embeddings = get_embeddings()
        store, unit_titles, experiments = build_index_from_data(data_dir, embeddings)

        index_dir.mkdir(parents=True, exist_ok=True)
        store.save_local(index_dir)

        print("Index rebuilt")
        print(f"Data dir: {data_dir}")
        print(f"Index dir: {index_dir}")
        print(f"Units: {len(unit_titles)}")
        print(f"Experiments: {len(experiments)}")
    except Exception as exc:
        raise SystemExit(f"Failed to rebuild index: {exc}") from exc


if __name__ == "__main__":
    main()
