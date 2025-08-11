#!/usr/bin/env python

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import random
import wikipedia
from langchain_anthropic import ChatAnthropic


# Configuration

MODEL = "claude-3-5-haiku-20241022"
# MODEL = "claude-sonnet-4-20250514"

llm = ChatAnthropic(model=MODEL)

OUTPUT_FILE = Path("./input/wikipedia_questions.jsonl")

PROMPT = """
Based on the following Wikipedia article about "{page_title}",
generate ONE interesting factual question and its answer.
The question should be about a specific fact mentioned in the article.
Do not choose questions whose answers are estimates or approximately known.

Currently saved question and answers for inspiration:
{reference_questions}

Article content:
{content}

Please respond in the following JSON format:
{{
    "question": "Your generated question here",
    "answer": "The specific answer based on the article content"
}}

Make sure the question is specific and the answer can be directly found in the provided content.
"""

QUESTIONS = []
if OUTPUT_FILE.is_file():
    with OUTPUT_FILE.open("r") as f:
        for line in f:
            question = json.loads(line)
            del question["source"]
            del question["timestamp"]
            QUESTIONS.append(question)


def generate_question(interactive=False):
    print("🔍 Getting random Wikipedia page...")

    while True:
        try:
            page = wikipedia.page(wikipedia.random())
            break
        except (wikipedia.DisambiguationError, wikipedia.PageError):
            continue

    print(f"📄 Found page: {page.title}")
    print(f"🔗 URL: {page.url}")
    print(f" Size: {len(page.content)} characters")

    if interactive:
        print("🦊 Opening page in Firefox...")
        subprocess.run(["firefox", "--new-tab", page.url])

    # Generate question and answer
    print("🤖 Generating question and answer using LLM...")

    try:
        questions = json.dumps(
            random.sample(QUESTIONS, min(len(QUESTIONS), 100)),
            indent=2,
            ensure_ascii=False,
        )

        response = llm.invoke(
            PROMPT.format(
                page_title=page.title,
                content=page.content,
                reference_questions=questions,
            ),
            temperature=1,
            max_tokens=512,
        )

        response_text = response.content.strip()

        qa_data = json.loads(response_text)

        question = qa_data.get("question")
        answer = qa_data.get("answer")

        if not question or not answer:
            print("❌ Failed to generate question and answer")
            return

    except Exception as e:
        print(f"Error generating question and answer: {e}")
        return

    question = question
    answer = answer

    # Display question and answer
    print("\n" + "=" * 50)
    print("GENERATED QUESTION AND ANSWER")
    print("=" * 50)
    print(f"📝 Question: {question}")
    print(f"💡 Answer: {answer}")
    print(f"🔗 Source: {page.url}")
    print("=" * 50)

    data = {
        "question": question,
        "answer": answer,
        "source": page.url,
        "timestamp": datetime.now().isoformat(),
    }

    QUESTIONS.append(
        {
            "question": question,
            "answer": answer,
        }
    )

    # Ask user for acceptance
    if not interactive or input("\nAccept question-answer? (Y/n): ").lower().strip() != "n":
        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        print(f"✅ Saved to {OUTPUT_FILE}")
    else:
        print("❌ Question-answer pair declined.")


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="Generate Wikipedia questions and answers.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ask to accept each question and answer.",
    )
    parser.add_argument(
        "amount",
        type=int,
        help="Number of questions to generate.",
    )
    args = parser.parse_args()

    print(f"🤖 Using model: {MODEL}")
    for _ in range(args.amount):
        generate_question(interactive=args.interactive)
