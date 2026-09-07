"""Three guardrail layers, cheapest first."""
import dspy


def validate_citations(pred):
    """Deterministic, free, catches fabricated citations outright."""
    valid = {d["chunk_id"] for d in pred.retrieved}
    fabricated = [c for c in (pred.citations or []) if c not in valid]
    return len(fabricated) == 0, fabricated


def should_abstain(docs, threshold):
    return (not docs) or docs[0].get("score", 0.0) < threshold


def temporal_conflict(docs):
    """Fires when retrieved chunks discuss the same statutory topic (shared
    Section/Rule/Form identifiers) but carry different effective dates.

    This is the failure RAGAS faithfulness cannot see: an answer quoting the
    pre-2019 registration threshold is faithful to a retrieved passage and
    still wrong in the world."""
    topics = [set(d["identifiers"]["sections"] + d["identifiers"]["rules"]
                  + d["identifiers"]["forms"]) for d in docs]
    overlapping = any(topics[i] & topics[j]
                      for i in range(len(topics)) for j in range(i + 1, len(topics)))
    dates = {d for doc in docs for d in doc["effective_dates"]}
    return overlapping and len(dates) > 1


class TemporalAnswer(dspy.Signature):
    """The excerpts contain multiple versions of the same provision from
    different dates. State the currently applicable position, name its
    effective date, and say explicitly what it replaced."""

    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField()
    effective_date: str = dspy.OutputField()
    superseded: str = dspy.OutputField(desc="the earlier position this replaced")


class Guarded(dspy.Module):
    def __init__(self, program, threshold=0.0):
        super().__init__()
        self.program, self.threshold = program, threshold
        self.temporal = dspy.ChainOfThought(TemporalAnswer)

    def forward(self, question, **kw):
        pred = self.program(question=question, **kw)
        pred.citations_ok, pred.fabricated = validate_citations(pred)
        pred.temporal_flag = temporal_conflict(pred.retrieved)
        if pred.temporal_flag:
            t = self.temporal(question=question, context=pred.context)
            pred.answer = (f"{t.answer}\n\nEffective date: {t.effective_date}. "
                           f"Supersedes: {t.superseded}")
            pred.effective_date = t.effective_date
        if should_abstain(pred.retrieved, self.threshold):
            pred.sufficient = False
            pred.answer = "The provided document does not contain enough information to answer this."
        return pred

