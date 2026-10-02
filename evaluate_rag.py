# ============================================================
# RAG CHATBOT - EVALUATION TEST
# ============================================================

import os


# ------------------------------------------------------------
# Test configuration
# ------------------------------------------------------------

APP_FILE = "app.py"

FAISS_FOLDER = "faiss_indexes"

DATA_FOLDER = "data"


# ------------------------------------------------------------
# Evaluation checklist
# ------------------------------------------------------------

tests = [
    {
        "name": "Application file exists",
        "condition": os.path.exists(APP_FILE)
    },
    {
        "name": "Data folder exists",
        "condition": os.path.exists(DATA_FOLDER)
    },
    {
        "name": "FAISS database folder exists",
        "condition": os.path.exists(FAISS_FOLDER)
    },
]


# ------------------------------------------------------------
# Display results
# ------------------------------------------------------------

print("=" * 60)

print("RAG CHATBOT EVALUATION")

print("=" * 60)

passed = 0

failed = 0


for test in tests:

    if test["condition"]:

        print(
            f"[PASS] {test['name']}"
        )

        passed += 1

    else:

        print(
            f"[FAIL] {test['name']}"
        )

        failed += 1


# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

print()

print("=" * 60)

print(
    f"Passed: {passed}"
)

print(
    f"Failed: {failed}"
)

print("=" * 60)


if failed == 0:

    print(
        "Basic project structure evaluation passed."
    )

else:

    print(
        "Some project structure tests failed."
    )