from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", cache_folder=".cache"
)


def read_ten_commandments(
    path: Path = Path("data/text/ten_commandments.txt"),
) -> list[str]:
    """Read ten commandments from file and returns it as list of strings."""
    sentences = []
    with open(path, "r", encoding="UTF-8") as infile:
        for line in infile:
            sentences.append(line.removesuffix("\n"))

    return sentences


def calculate_embeddings(sentences: list[str]) -> list[list[float]]:
    embeddings = model.encode(sentences)
    return embeddings


def calculate_single_embedding(sentence: str) -> list[float]:
    embeddings = model.encode([sentence])
    return embeddings


def create_flat_index(
    embeddings: list[list[float]], dim: int = 384
) -> faiss.IndexFlatL2:
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)  # pylint: disable=no-value-for-parameter

    return index


def store_index(index: faiss.IndexFlatL2, path: Path = Path("faiss.db")) -> None:
    faiss.write_index(index, str(path))


def read_index(path: Path = Path("faiss.db")) -> faiss.IndexFlatL2:
    return faiss.read_index(str(path))
