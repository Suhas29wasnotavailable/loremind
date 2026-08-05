"""Calibrate GROUNDING_MIN_SCORE and the candidate pool size.

Measures where the cross-encoder's top score separates answerable questions from
unanswerable ones, so the threshold is chosen from data rather than guessed.
"""
import logging

from backend.logging_setup import setup_logging

setup_logging(logging.WARNING)

from backend import config
from backend.retrieval import get_retriever

# Questions the corpus genuinely answers, phrased the way a user would.
ON_CORPUS = [
    "Who is Darth Vader?",
    "How did Anakin fall to the dark side?",
    "What was Order 66?",
    "Tell me about the Death Star",
    "How did Yoda die?",
    "Who trained Luke Skywalker?",
    "What is the Rule of Two?",
    "Who was Ahsoka Tano?",
    "What happened at the Battle of Endor?",
    "What are midi-chlorians?",
    "Who is Din Djarin?",
    "What is the Darksaber?",
    "How did Obi-Wan defeat Anakin?",
    "What is a Force dyad?",
    "Who built C-3PO?",
    "What was the Inquisitorius?",
    "Tell me about Palpatine's rise to power",
    "What is kyber crystal used for?",
    "Who was Jango Fett?",
    "What happened to the Jedi Temple?",
    "Why did Luke go into exile?",
    "What is Exegol?",
]

# Questions the corpus cannot answer. These must refuse.
OFF_CORPUS = [
    "What is the capital of France?",
    "Explain how to write a Python decorator",
    "Who was Grand Admiral Thrawn's flagship commander?",
    "What is the airspeed velocity of an unladen swallow?",
    "Give me a recipe for carbonara",
    "What did Thrawn collect as a hobby?",
    "Who won the 2019 Super Bowl?",
    "How do I reset my email password?",
    "What is the population of Tokyo?",
    "Summarise the plot of The Great Gatsby",
]

r = get_retriever()
print(f"pool: semantic={config.SEMANTIC_TOP_K} bm25={config.BM25_TOP_K}\n")


def top_scores(queries):
    out = []
    for q in queries:
        _, sources, _ = r.retrieve(q)
        out.append((q, sources[0]["score"] if sources else -99.0))
    return out


on = top_scores(ON_CORPUS)
off = top_scores(OFF_CORPUS)

print("ON-CORPUS (want high scores)")
for q, s in sorted(on, key=lambda x: x[1]):
    print(f"  {s:7.2f}  {q}")

print("\nOFF-CORPUS (want low scores)")
for q, s in sorted(off, key=lambda x: x[1], reverse=True):
    print(f"  {s:7.2f}  {q}")

on_s = sorted(s for _, s in on)
off_s = sorted((s for _, s in off), reverse=True)
print(f"\non-corpus  min={on_s[0]:.2f}  p10={on_s[len(on_s)//10]:.2f}  median={on_s[len(on_s)//2]:.2f}")
print(f"off-corpus max={off_s[0]:.2f}  median={sorted(s for _,s in off)[len(off_s)//2]:.2f}")

print("\nthreshold sweep:")
print(f"  {'thresh':>7}  {'answered':>8}  {'refused-off':>11}  {'note'}")
for t in (-9, -8, -7, -6, -5, -4, -3, -2, -1, 0):
    recall = sum(1 for _, s in on if s >= t) / len(on)
    blocked = sum(1 for _, s in off if s < t) / len(off)
    print(f"  {t:7.1f}  {recall:7.0%}  {blocked:10.0%}   {'both perfect' if recall==1 and blocked==1 else ''}")

r.client.close()
