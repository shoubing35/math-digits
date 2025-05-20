# Digit Matching with SFT on LLMs

This project fine-tunes an open-source language model to solve a symbolic reasoning problem:

> “A book with 50 pages, numbered 1 to 50, has its pages renumbered in reverse (page 1 becomes 50, page 2 becomes 49, etc.). How many pages retain the same ones digit before and after renumbering?”

It uses a dataset of 100 problem variations and applies a full SFT fine-tuning pipeline on `meta-llama/Llama-3.2-1B-Instruct` (RM and PPO are work-in-progress). The goal is to evaluate how well small LLMs generalize across paraphrasing and numerical variation when trained on a single reasoning task.

---

## Why This Matters

LLMs often struggle with consistent symbolic reasoning—especially when small input changes cause failures in setup or execution. This project:

- Tests LLM generalization across prompt phrasing and number changes
- Explores how well SFT improve reasoning consistency (RLHF is work-in-progress)
- Analyzes failure types: incorrect modulo setup, computation errors, faulty counting

---

## Repo Structure

- data/ # Dataset generation and CSV
- training/ # SFT and PPO training scripts
- eval/ # Evaluation logic and scoring
- upload / # Upload dataset to HF
- trl_custom/ # Optional: trainer overrides

## Quickstart
- Colab: https://colab.research.google.com/drive/1lnjfql_zdhYa4HXQmsCoTtNffjArq8W-?usp=drive_link
- Generate dataset, run SFT training, compare inference results between base model and SFT-trained model

## Results
Model	Accuracy on Validation Set:
- Base (no fine-tuning)	0/10
- SFT (LoRA, 4 epochs)	5/10
- PPO (work-in-progress)

Failure Types:
- Incorrect congruence setup
- Partial arithmetic errors
- Incorrect digit frequency counting

## Example Completions
### Prompt
"You are a math expert with clear and concise reasoning. Solve this problem step-by-step and box your final numerical answer:"
“A book with 50 pages, numbered 1 to 50, has its pages renumbered in reverse (page 1 becomes 50, page 2 becomes 49, etc.). How many pages retain the same ones digit before and after renumbering?”
### Correct Output
1. Define the renumbering.
Original page n (1 ≤ n ≤ 113) is renumbered to N = 114 - n.

2. Set the ones-digit condition.
Require n mod 10 = N mod 10:
n mod 10 = (114 - n) mod 10.

3. Translate into a congruence.
Let x = n mod 10. Then:
x ≡ (114 - n) mod 10 = (114 mod 10) - x mod 10 = 4 - x (mod 10),
so 2x ≡ 4 (mod 10).

4. Solve the congruence.
Solutions are x ∈ [2, 7].

5. Count pages ending with each solutions digit and sum:
Pages ending in 2: 12
Pages ending in 7: 11
Total = 23.

\boxed{23}

### Incorrect Output
- Logical error in step 3

Preserve the count of pages with the same ones digit.

Step 1: Identify the transformation
Each page number \( n \) is transformed to \( 51 - n \).
We want to find how many values of \( n \in [1, 50] \) satisfy:
\[ n \mod 10 = (51 - n) \mod 10 \]

Step 2: Define the condition using modular arithmetic
Let \( x = n \mod 10 \). Then:
\[ x = (51 - n) \mod 10 = (1 - x) \mod 10 \]
\[ 2x \equiv 1 \mod 10 \]

Step 3: Solve the congruence
We solve for integer \( x \) such that:
\[ 2x \equiv 1 \mod 10 \]
Check all values of \( x = 0 \) to \( 9 \) to see if any satisfy this.

A valid solution exists: \( x = 5 \), so we count how many pages end in 5.
That occurs 6 times in the range 1 to 50.

Step 4: Provide the final answer
The final answer is: $\boxed{6}$

## Dataset
100 hand-generated examples:
- Varying book lengths from 30–120 pages
- Prompt paraphrasing and rewordings
- Each example includes boxed final answer

Dataset available at:
[shoubing35/ones_digit_sft_dataset](https://huggingface.co/datasets/shoubing35/ones_digit_sft_dataset)

## About
Built and maintained by Charles Chen.
This repo is a showcase of reasoning generalization using reward-model-driven fine-tuning of LLMs.

Interested in:
- Designing AI for creative problem solving (math reasoning, music generation, etc.)

Feel free to connect!
