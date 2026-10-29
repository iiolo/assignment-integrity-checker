# assignment-integrity-checker

A local tool that helps teachers check student assignments submitted to Teams (Education)
for plagiarism and AI-generated content. It runs entirely on the teacher's own machine as a
Streamlit web UI, no separate server or hosting required.

## Features

- **Text extraction** (`src/extract_text.py`) — extracts text from `.docx` and `.pdf` files
- **Similarity check** (`src/similarity.py`) — compares every pair of student submissions
  using word 5-gram containment plus a paraphrase check, and flags pairs above a configurable
  threshold (see [How similarity is calculated](#how-similarity-is-calculated))
- **AI-generated text detection** (`src/ai_detection_check.py`) — runs a local RoBERTa model
  (`fakespot-ai/roberta-base-ai-text-detection-v1`, Apache-2.0) to estimate the probability
  that a submission was AI-generated
- **Streamlit UI** (`streamlit_app.py`) — upload student files, run both checks, and view the
  results in one report

## Current status

- Text extraction, similarity check, and AI detection are complete and working.
- Streamlit UI works with uploaded files and with automatic fetching from Teams (Microsoft
  sign-in via device code, `src/teams_auth.py`; submissions and the assignment's attached
  template via the Graph Education API, `src/teams_client.py`).
- Packaged as a standalone `.exe` with PyInstaller (`assignment_checker.spec`).

## How similarity is calculated

Honest answers to the same question share vocabulary, so plain word-frequency measures
(e.g. TF-IDF cosine) score almost every pair of same-topic answers as "similar". Instead:

1. **Template removal.** If the teacher's blank template or question sheet is available, its
   text is removed from every submission first (`strip_template`). On Teams it is fetched from
   the assignment's attached files; otherwise it can be uploaded. Without a template, lines and
   sentences shared by most students are detected and removed automatically.
2. **Word 5-gram containment.** Each text is split into overlapping runs of 5 words
   (shingles). The score is the share of the *shorter* text's shingles that also appear in the
   other text, so copying a whole answer from a longer one is still caught. Wording shared by
   many students, URLs, and texts with too few shingles are ignored.
3. **Copied-passage evidence.** Overlapping shared shingles are merged into the longest runs of
   identical wording and shown as evidence (shorter runs are dropped).
4. **Paraphrase check.** The share of the shorter text's sentences that have a near-identical
   meaning in the other text (sentence embeddings, `all-MiniLM-L6-v2`) catches reworded copying.

Excluded text is never hidden: the report lists exactly which template or shared lines were
removed, so the teacher can check that no real answer was dropped (fully automatic removal of
common segments is known to produce false results without human validation).

The final score is the higher of the containment score (2) and the paraphrase score (4). The default alert
threshold is 10%; honest answers to the same question typically score under 5%. The threshold was
calibrated on a single class, so adjust it using the evidence shown in the report.

### AI-text detection

Each submission is scored by a local RoBERTa classifier. Several design choices follow the
survey by Fraser, Dawkins & Kiritchenko (JAIR 2025):

- **Enough context per score.** Detectors need roughly 100-200 words to reach their potential and
  lose accuracy on sentence-length input, so a student's consecutive paragraphs are merged into
  chunks of about 150 words before scoring, and a submission with fewer than 100 words of the
  student's own text is reported as "Too short" instead of being graded (an empty text can
  otherwise score as "AI").
- **Class-relative grading.** Detectors misclassify writing by non-native English speakers as
  AI-generated far more often than native writing (Liang et al., 2023), so each student is graded
  against their own class's score distribution and only students well above their classmates are
  flagged.
- **Not a replacement for a conversation.** Teachers cannot reliably spot AI-written essays by
  reading them either: in two experiments with pre-service (N = 89) and experienced teachers
  (N = 200), only 45% and 38% of AI essays were identified, while confidence stayed around 80%,
  and AI text deliberately written at a lower level was the hardest to recognise (Fleckenstein et
  al., 2024). The report therefore points to evidence rather than impressions and recommends asking
  the student to explain their work in person as the next step.
- **Limits that are reported, not hidden.** Text a person wrote and then had AI reword or polish
  is the hardest case (detectors can perform worse than chance), and a classifier tuned on one
  kind of text generalises poorly to others. A "Low" grade therefore never proves a student wrote
  everything themselves.

#### Evaluation on a labelled corpus

The AI detector was checked locally against the human/AI text corpus
[LorenzM97/human-AI-generatedTextCorpus](https://github.com/LorenzM97/human-AI-generatedTextCorpus)
(English Wikipedia and news; 100 human texts per domain and 400 GPT-3.5 texts generated with four
prompt styles). The corpus shows no license, so it is not included in this repository.

Ranking quality (AUC, 1.0 = perfect; random subsets of 40 human and 80 AI texts):

| Texts | Per-paragraph scoring (before) | 150-word chunks (now) |
| --- | --- | --- |
| Wikipedia, full text | 0.917 | 0.939 |
| Wikipedia, first 200 words | 0.874 | 0.885 |
| News, full text | 0.989 | 0.997 |
| News, first 200 words | 0.782 | 0.941 |

Simulated classes of 20 (random draws from the corpus, graded with `grade_ai_scores`):

| Class | Human students wrongly graded Medium/High | AI students graded Medium/High |
| --- | --- | --- |
| Wikipedia-style, 0 AI | 35% | - |
| Wikipedia-style, 2 AI | 28% | 87% |
| Wikipedia-style, 5 AI | 11% | 48% |
| Wikipedia-style, 10 AI | 0% | 0% |
| News, 0 AI | 16% | - |
| News, 2 AI | 16% | 100% |
| News, 5 AI | 15% | 100% |
| News, 10 AI | 0% | 17% |

What this means for use:

- Chunking improves ranking, but absolute scores are not trustworthy on their own: on formal,
  encyclopedic human writing a fixed cutoff of 60 flags 40-78% of human texts.
- Relative grading still flags some human students in a class that uses no AI at all, and
  formal writing is the worst case. Widening the margin by the class's spread removes those false
  flags but also removes almost all true detections (e.g. Wikipedia-style class: 4% false vs 11%
  caught), so the margin was left unchanged. Treat Medium/High as a reason to look at the passages
  and talk to the student, never as a finding.
- When about half the class or more uses AI, the class median itself rises and nobody stands out.
- Limits of this check: one generator (GPT-3.5), Wikipedia/news rather than student work, and the
  human news texts are about twice as long as the AI ones (the "first 200 words" rows remove that
  length difference).

### References

Methods used:

- Broder, A. Z. *On the resemblance and containment of documents.* Compression and Complexity
  of Sequences, 1997 — shingling and the containment measure.
- Schleimer, S., Wilkerson, D. S., Aiken, A. *Winnowing: Local Algorithms for Document
  Fingerprinting.* ACM SIGMOD, 2003.
  [doi:10.1145/872757.872770](https://doi.org/10.1145/872757.872770) — k-gram fingerprinting as
  used by MOSS (this project does not implement winnowing; it compares all word 5-grams directly).
- PAN text-alignment shared tasks (CLEF) — the seeding / extension / filtering pipeline for
  finding reused passages (e.g. Sanchez-Perez, Gelbukh, Sidorov, PAN 2014).
- Clough, P., Stevenson, M. *Developing a corpus of plagiarised short answers.* Language
  Resources and Evaluation 45(1), 2011.
  [doi:10.1007/s10579-009-9112-1](https://doi.org/10.1007/s10579-009-9112-1) — plagiarism in short
  student answers.
- Liang, W., Yuksekgonul, M., Mao, Y., Wu, E., Zou, J. *GPT detectors are biased against
  non-native English writers.* Patterns, 2023.
  [doi:10.1016/j.patter.2023.100779](https://doi.org/10.1016/j.patter.2023.100779) — why AI scores
  are graded relative to the class.
- Fraser, K. C., Dawkins, H., Kiritchenko, S. *Detecting AI-Generated Text: Factors Influencing
  Detectability with Current Methods.* Journal of Artificial Intelligence Research 82, 2025.
  [link](https://www.jair.org/index.php/jair/article/view/16665/27172) — minimum text length,
  chunking, and the limits of detecting polished or paraphrased text.
- Fleckenstein, J., Meyer, J., Jansen, T., Keller, S. D., Koller, O., Moller, J. *Do teachers spot
  AI? Evaluating the detectability of AI-generated texts among student essays.* Computers and
  Education: Artificial Intelligence 6, 2024.
  [doi:10.1016/j.caeai.2024.100209](https://doi.org/10.1016/j.caeai.2024.100209) — why the report
  recommends a conversation with the student instead of relying on a teacher's impression.

Data used for evaluation:

- LorenzM97, [*human-AI-generatedTextCorpus*](https://github.com/LorenzM97/human-AI-generatedTextCorpus)
  (GitHub; no paper or license listed) — labelled human and GPT-3.5 texts used to check the AI
  detector.

Recent work consulted (informed the design, not implemented as-is):

- Amirzhanov, A. *Plagiarism types and detection methods: a systematic survey of algorithms in
  text analysis.* Frontiers in Computer Science, 2025.
  [doi:10.3389/fcomp.2025.1504725](https://doi.org/10.3389/fcomp.2025.1504725)
- *Overview of the Plagiarism Detection Task at PAN 2025.*
  [arXiv:2510.06805](https://arxiv.org/abs/2510.06805) — its baselines combine embedding
  similarity with TF-IDF, in line with this tool's hybrid lexical + semantic design (that task
  targets machine-generated plagiarism in scientific articles).
- Tang, Hu, Han. [*Efficient Plagiarism Detection via Sentence Embeddings*](https://www.dei.unipd.it/~faggioli/temp/clef2025/paper_324.pdf)
  (CLEF 2025 notebook) — reports low recall for `all-MiniLM-L6-v2` in a retrieval setting and
  prefers `e5-base-v2`. Here embeddings are only a secondary paraphrase signal with a high
  threshold, but a larger embedding model is a possible future improvement.

Models used: [`fakespot-ai/roberta-base-ai-text-detection-v1`](https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1)
(AI-text detection) and [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
(paraphrase check), both Apache-2.0.

### Attribution and licensing

This repository only cites and links to the works above and describes their ideas in its own
words; it contains no text, figures or data copied from them. The evaluation corpus was
downloaded for local testing only and is not redistributed here; only aggregate results are
reported. If you reuse this project, check the license of each model and dataset you rely on.

## Setup

```
pip install -r requirements.txt
```

If you plan to use the Teams integration, create a `.env` file in the project root with:

```
AZURE_CLIENT_ID=<your Azure AD app's client ID>
AZURE_TENANT_ID=<your Azure AD tenant ID>
```

## Running the app

```
streamlit run streamlit_app.py
```

## Testing the Teams login flow

```
python src/teams_auth.py
```

This starts an MSAL device code sign-in flow. It requires a signed-in Microsoft 365 account
in the target tenant, and admin consent for `EduAssignments.ReadBasic`.

## Note

AI-detection and similarity results are for reference only and should not be used as the
sole basis for a plagiarism or AI-use decision.
