from itertools import combinations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarities(
    student_texts: dict[str, str], threshold: float = 40.0
) -> list[tuple[str, str, float]]:
    """Compare every pair of students' texts and return pairs with similarity >= threshold(%)."""
    students = list(student_texts.keys())

    # No comparison needed with fewer than 2 students
    if len(students) < 2:
        return []

    # Vectorize all students' texts using TF-IDF
    texts = [student_texts[name] for name in students]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Compute the cosine similarity matrix between all document pairs
    similarity_matrix = cosine_similarity(tfidf_matrix)

    results: list[tuple[str, str, float]] = []

    # Iterate over every unique student pair (A-B) and keep only pairs at or above the threshold
    for i, j in combinations(range(len(students)), 2):
        score = similarity_matrix[i][j] * 100
        if score >= threshold:
            results.append((students[i], students[j], score))

    return results
