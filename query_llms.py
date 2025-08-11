#!/usr/bin/env python

import asyncio
import itertools as it
import json
import re
import time
from datetime import timedelta
from pathlib import Path

from together import AsyncTogether

EXPERIMENT = "base"

JUDGE_MODEL = "openai/gpt-oss-120b"

QUESTIONS = Path("./input") / "wikipedia_questions.jsonl"
SYSTEM = Path("./system") / EXPERIMENT
MODELS = Path("./input") / "models.jsonl"

OUTPUT = Path("./output") / f"{EXPERIMENT}.jsonl"

ASYNC_CLIENT = AsyncTogether()

CONCURRENCY_LIMIT = 8
SEMAPHORE = asyncio.Semaphore(CONCURRENCY_LIMIT)

EVALUATE_SYSTEM = (Path("./system") / "evaluate.txt").read_text()

EVALUATE_PROMPT = """
Question: {question}
Received Answer: {received_answer}
Expected Answer: {expected_answer}
"""

CATEGORIES = {
    "CORRECT": "✅",
    "INCORRECT": "❌",
    "DOUBT": "⬜",
    "ERROR": "❓",
}


async def query_model(model: str, prompt: str, system: str = "") -> str:
    """Query model with given prompt and question."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = await ASYNC_CLIENT.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.01,
    )
    return response.choices[0].message.content


def extract_answer(response: str) -> str:
    """Parse the model response to find the answer."""
    lines = response.strip().split("\n")

    # Look for lines starting with "Answer:" (case insensitive)
    for line in reversed(lines):
        line = line.strip()
        match = re.search(r".*answer:\s*\{?(.*)?\}?", line, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # fallback to returning full response and let judge decide
    return response


def extract_evaluation(response: str) -> str:
    """Parse the model response to find the category."""
    lines = response.strip().split("\n")

    # Look for lines starting with "Category:" (case insensitive)
    evaluation = ""
    for line in reversed(lines):
        line = line.strip()
        match = re.search(r".*category:\s*\{?([a-zA-Z]+)\}?\.?", line, re.IGNORECASE)
        if match:
            evaluation = match.group(1).strip()

    if evaluation not in CATEGORIES:
        raise ValueError(f"No valid category found in the response:\n{response}")

    return evaluation


async def process_combination(
    question: dict, system_type: str, system: str, model_name: str
) -> dict | None:
    """Process a single question/prompt/model combination and return result dictionary."""
    async with SEMAPHORE:
        try:
            model_response = await query_model(
                model_name,
                f"Question: {question['question']}?",
                system=system,
            )
            received_answer = extract_answer(model_response)

            received_evaluation = await query_model(
                JUDGE_MODEL,
                prompt=EVALUATE_PROMPT.format(
                    question=question["question"],
                    expected_answer=question["answer"],
                    received_answer=received_answer,
                ),
                system=EVALUATE_SYSTEM,
            )
            evaluation = extract_evaluation(received_evaluation)

            result = {
                "question": question["question"],
                "expected_answer": question["answer"],
                "model": model_name,
                "suggest_empty": system_type == "suggest_empty",
                "response": model_response,
                "received_answer": received_answer,
                "evaluation": evaluation,
            }

            print(f"{system_type} - {model_name} - {question['question']}?"[:140])
            print(
                f"  Expected: {question['answer']}\n"
                + f"{CATEGORIES[evaluation]}Got:      {received_answer}"
            )

            if evaluation == "ERROR":
                return None

            return result

        except Exception as e:
            print(f"{system_type} - {model_name} - {question['question']}?"[:140])
            print(f"🚨 EXCEPTION: {e}")
            return None


async def main():
    with QUESTIONS.open() as f:
        questions = [json.loads(line) for line in f]

    system = {
        "base": (SYSTEM / "base.txt").read_text(),
        "suggest_empty": (SYSTEM / "empty.txt").read_text(),
    }

    with MODELS.open() as f:
        models = [json.loads(line)["name"] for line in f]

    if EXPERIMENT == "test":
        questions = questions[:4]

    print(f"Starting experiment: {EXPERIMENT}")
    print(f"Questions: {len(questions)}")
    print(f"Models: {len(models)}")
    print(f"System Prompts: {list(system.keys())}")
    total_combinations = len(questions) * len(system) * len(models)
    print(f"Total combinations: {total_combinations}")

    # load saved results to check what's already done
    if OUTPUT.is_file():
        with OUTPUT.open() as f:
            existing_results = {
                ((result := json.loads(line))["question"], result["model"], result["suggest_empty"])
                for line in f
            }
    else:
        existing_results = set()
        OUTPUT.touch()

    print(f"Already completed: {len(existing_results)} combinations")
    remaining = total_combinations - len(existing_results)
    print(f"Remaining: {remaining} combinations\n")

    # Create tasks for all remaining combinations
    tasks = []
    for question, (system_type, system), model_name in it.product(
        questions, system.items(), models
    ):
        suggest_empty = system_type == "suggest_empty"
        if (question["question"], model_name, suggest_empty) in existing_results:
            continue

        task = process_combination(question, system_type, system, model_name)
        tasks.append(task)

    print(f"Starting {len(tasks)} async tasks with concurrency limit of {CONCURRENCY_LIMIT}")
    start_time = time.time()

    # Run all tasks concurrently with progress updates
    successful_results = []
    completed_count = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        completed_count += 1

        if result is not None:
            successful_results.append(result)

        if completed_count % 50 == 0:
            elapsed = time.time() - start_time
            rate = completed_count / elapsed if elapsed > 0 else 0
            remaining_tasks = len(tasks) - completed_count
            eta_seconds = remaining_tasks / rate if rate > 0 else 0

            eta_formatted = str(timedelta(seconds=int(eta_seconds)))

            progress_pct = (completed_count / len(tasks)) * 100
            print(f"\n  Progress: {completed_count}/{len(tasks)} ({progress_pct:.1f}%)")
            print(f"  ETA: {eta_formatted}\n")

            with OUTPUT.open("a") as f:
                for result in reversed(successful_results):
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    successful_results.pop()

    print(f"\nCompleted {len(successful_results)} out of {len(tasks)} tasks")

    with OUTPUT.open("a") as f:
        for result in successful_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    elapsed_time = time.time() - start_time
    elapsed_formatted = str(timedelta(seconds=int(elapsed_time)))
    print(f"Total time: {elapsed_formatted}")


if __name__ == "__main__":
    asyncio.run(main())
    print("All combinations completed!")
