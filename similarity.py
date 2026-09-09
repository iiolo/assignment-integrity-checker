from itertools import combinations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarities(
    student_texts: dict[str, str], threshold: float = 40.0
) -> list[tuple[str, str, float]]:
    """학생별 텍스트를 서로 비교해서 유사도가 threshold(%) 이상인 쌍만 반환한다."""
    students = list(student_texts.keys())

    # 비교할 학생이 2명 미만이면 계산할 필요가 없음
    if len(students) < 2:
        return []

    # TF-IDF로 모든 학생의 텍스트를 벡터화
    texts = [student_texts[name] for name in students]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # 모든 문서 쌍 간의 코사인 유사도 행렬 계산
    similarity_matrix = cosine_similarity(tfidf_matrix)

    results: list[tuple[str, str, float]] = []

    # 중복 없이 모든 학생 쌍(A-B)을 순회하며 임계값 이상인 쌍만 결과에 추가
    for i, j in combinations(range(len(students)), 2):
        score = similarity_matrix[i][j] * 100
        if score >= threshold:
            results.append((students[i], students[j], score))

    return results
