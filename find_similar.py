import numpy as np
from utils import read_index, read_ten_commandments, calculate_single_embedding

if __name__ == "__main__":
    index = read_index()
    sentences = read_ten_commandments()

    prompt = input("Please provide a search term:\n")
    embedding = calculate_single_embedding(prompt)

    probs, results = index.search(np.array(embedding), 3)
    probs, results = probs[0], results[0]

    print()
    print("Here are the top three results (lower score = better match):")

    for prob, result in zip(probs, results):
        print(f"Score: {prob:.3f} - {sentences[result]}")
