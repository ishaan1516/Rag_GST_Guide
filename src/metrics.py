"""Retrieval metrics + LLM-judge answer metrics."""
import numpy as np
import dspy

import config


def recall_at_k(retrieved_ids, gold_ids, k=5):
    if not gold_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & set(gold_ids)) / len(gold_ids)


def mrr(retrieved_ids, gold_ids):
    for i, cid in enumerate(retrieved_ids):
        if cid in gold_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids, gold_ids, k=5):
    dcg = sum(1 / np.log2(i + 2) for i, c in enumerate(retrieved_ids[:k]) if c in gold_ids)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(gold_ids))))
    return dcg / idcg if idcg else 0.0


class Judge(dspy.Signature):
    """Grade a GST answer against the gold answer and the retrieved context."""

    question: str = dspy.InputField()
    gold_answer: str = dspy.InputField()
    context: str = dspy.InputField()
    prediction: str = dspy.InputField()
    correct: bool = dspy.OutputField(desc="factually equivalent to the gold answer")
    grounded: bool = dspy.OutputField(desc="every claim is supported by the context")
    reasoning: str = dspy.OutputField(desc="one sentence: what specifically was wrong")


_judge = None


def get_judge(api_key=None):
    """Judge runs on a DIFFERENT model from the generator - a model grading
    its own output is a known bias. This holds on OpenRouter too: pick two
    different free model families, not two variants of the same one, and
    spot-check a handful of verdicts by hand - small free judges are noisier
    than a paid model."""
    global _judge
    if _judge is None:
        if config.PROVIDER == "openrouter":
            lm = dspy.LM(config.OPENROUTER_JUDGE_MODEL,
                         api_key=api_key or config.OPENROUTER_API_KEY,
                         temperature=0.0, max_tokens=1500)
        else:
            lm = dspy.LM(config.JUDGE_MODEL, api_key=api_key or config.OPENAI_API_KEY,
                         temperature=0.0, max_tokens=1500)
        _judge = dspy.ChainOfThought(Judge)
        _judge.set_lm(lm)
    return _judge


def judge_safely(question, gold_answer, context, prediction, retries=1):
    """The free-tier judge occasionally returns something DSPy's adapters
    can't parse - a reasoning model that burns its whole budget without
    finishing, or a stray safety-classifier model that ignores the task
    entirely. Retry once (the auto-router may land on a different
    underlying model), then return None rather than let one bad row crash
    the whole evaluation run."""
    for attempt in range(retries + 1):
        try:
            return get_judge()(question=question, gold_answer=gold_answer,
                               context=context, prediction=prediction)
        except Exception as e:
            if attempt == retries:
                print(f"  [warn] judge failed after {retries + 1} attempts "
                      f"({type(e).__name__}) - row scored as unjudged")
                return None
    return None


def citations_valid(pred):
    valid = {d["chunk_id"] for d in pred.retrieved}
    return bool(pred.citations) and all(c in valid for c in pred.citations)


def composite_metric(example, pred, trace=None):
    if not getattr(example, "answerable", True):
        return 1.0 if not pred.sufficient else 0.0      # hard zero for answering the unanswerable
    j = judge_safely(example.question, example.gold_answer, pred.context, pred.answer)
    if j is None:
        return 0.0  # unjudged - score conservatively rather than skip
    return (0.5 * float(j.correct) + 0.3 * float(j.grounded)
            + 0.2 * float(citations_valid(pred)))


def gepa_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA mutates prompts by reflecting on the feedback string, so the
    feedback is the entire mechanism. 'incorrect' teaches it nothing."""
    if not getattr(example, "answerable", True):
        ok = not pred.sufficient
        return dspy.Prediction(score=1.0 if ok else 0.0,
                               feedback="Correctly abstained." if ok else
                               "Answered a question the document does not cover; "
                               "should have set sufficient=False.")
    j = judge_safely(example.question, example.gold_answer, pred.context, pred.answer)
    if j is None:
        return dspy.Prediction(score=0.0, feedback="Judge failed to return a "
                               "parseable verdict for this example.")
    score = (0.5 * float(j.correct) + 0.3 * float(j.grounded)
             + 0.2 * float(citations_valid(pred)))
    fb = "Correct and fully grounded." if score >= 1.0 else j.reasoning
    return dspy.Prediction(score=score, feedback=fb)