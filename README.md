# assignment-integrity-checker

A local tool that helps teachers check student assignments submitted to Teams (Education)
for plagiarism and AI-generated content. It runs entirely on the teacher's own machine as a
Streamlit web UI, no separate server or hosting required.

## Features

- **Text extraction** (`src/extract_text.py`) — extracts text from `.docx` and `.pdf` files
- **Similarity check** (`src/similarity.py`) — compares every pair of student submissions
  using TF-IDF + cosine similarity and flags pairs above a configurable threshold
- **AI-generated text detection** (`src/ai_detection_check.py`) — runs a local RoBERTa model
  (`fakespot-ai/roberta-base-ai-text-detection-v1`, Apache-2.0) to estimate the probability
  that a submission was AI-generated
- **Streamlit UI** (`streamlit_app.py`) — upload student files, run both checks, and view the
  results in one report

## Current status

- Text extraction, similarity check, and AI detection are complete and working.
- Streamlit UI is working for manually uploaded files.
- Automatic fetching of submissions from Teams via the Microsoft Graph Education API
  (`src/teams_auth.py`) is blocked on tenant admin consent for the `EduAssignments.ReadBasic`
  permission. Until that is granted, teachers must download files from Teams manually and
  upload them into the app.
- Packaging into a standalone `.exe` with PyInstaller has not started yet.

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
