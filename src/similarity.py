from itertools import combinations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# A line shared by at least this fraction of submissions is treated as shared template
# text (e.g. assignment instructions) rather than a sign of copying, and is stripped
# before comparison. Kept high so a small group of students copying each other isn't
# mistaken for everyone sharing a template.
COMMON_LINE_THRESHOLD = 0.7


def _strip_common_lines(student_texts: dict[str, str]) -> dict[str, str]:
    """Remove lines that most submissions share verbatim (likely a shared template/instructions)."""
    lines_by_student = {
        name: [line.strip() for line in text.splitlines() if line.strip()]
        for name, text in student_texts.items()
    }

    # Skip with too few submissions: not enough students to tell a shared template
    # apart from a small group copying each other
    if len(lines_by_student) < 3:
        return student_texts

    line_counts: dict[str, int] = {}
    for lines in lines_by_student.values():
        for line in set(lines):
            line_counts[line] = line_counts.get(line, 0) + 1

    n = len(lines_by_student)
    common_lines = {line for line, count in line_counts.items() if count / n >= COMMON_LINE_THRESHOLD}

    return {
        name: "\n".join(line for line in lines if line not in common_lines)
        for name, lines in lines_by_student.items()
    }


def calculate_similarities(
    student_texts: dict[str, str], threshold: float = 40.0
) -> list[tuple[str, str, float]]:
    """Compare every pair of students' texts and return pairs with similarity >= threshold(%)."""
    students = list(student_texts.keys())

    # No comparison needed with fewer than 2 students
    if len(students) < 2:
        return []

    student_texts = _strip_common_lines(student_texts)

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
