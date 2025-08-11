# Experiments to trigger uncertainty

The goal of the following experiments is to explore scenarios where the LLM might doubt its knowledge or reasoning abilities, by:

- Providing fake statistics that the model is performing badly.
- Gamifying the answer: Penalize mistakes, while allowing the easy way out of "I don't know".

## Stating success rate

Prompt:

```text
Out of X questions of similar difficulty, you answered correctly to Y of them so far.

Question X+1:

{question/math prob here}
```

We would test if the quality of the answer depends on X and Y.

## Scoring answers

Prompt:

```text
"Answer this question:

{question/math prob here}

You get the following points depending on your answer:

Correct: +1
Incorrect: N (N<1)

```

We did not state explicitly what the "goal" is. Two possible views are:

1. Try to answer the question correctly, disregarding the prompt above.
2. Try to maximize the score.

In this case they are equivalent, but the model's reasoning approach could show sensitivity to N.
Is it more conservative/vague as N decreases, for example?

### I don't know

We can extend this test to distinguish the two goals by scoring "Do not know" answers separately,
which would previously fall under "Incorrect".

```
Do not know/blank: B (N<=B<1)
```

Does the LLM change some of its answers to "Do not know"?
Is it able to improve its score?
Do we see behavior transitions on N and B?

Multiple choice questions would be easier to process and are a natural setting that the LLM would be familiar with.

### Performance context

We could further add a "current total score" to the prompt. Is the model riskier if the total score is much larger than what is lost on a wrong answer?

## Complete version of answer scoring experiment

```text
Question: {question}

You get the following points depending on your answer:

Correct: +1
Incorrect: N (N<1)
Blank: B (N<=B<1)

Your current total score: T

Write your answer at the end of your message in this format:

Answer: {answer or blank}
```

# Experiments on consistency

## Consistency of the unknown

Take a non-reasoning model. Ask it:

```text
Generate questions with verifiable answers that you do not know the answer to, possibly questions that could be answered with capabilities like extended thinking and web search.

Do not generate time-sensitive questions pertaining to the time period after your last training time.
```

Independently, we ask it to answer them.

