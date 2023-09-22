from utils import (
    calculate_embeddings,
    create_flat_index,
    read_ten_commandments,
    store_index,
)

if __name__ == "__main__":
    sentences = read_ten_commandments()
    embeddings = calculate_embeddings(sentences)
    index = create_flat_index(embeddings)
    store_index(index)
