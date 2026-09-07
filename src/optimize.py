"""Run a real DSPy optimizer and save the compiled program + prompt diff."""
import argparse, json
from pathlib import Path

import dspy
from dspy.teleprompt import MIPROv2, BootstrapFewShotWithRandomSearch

import config
from dspy_program import GSTQA, setup_lm
from evalset import to_dspy
from metrics import composite_metric, gepa_metric
from retrieve import Retriever


def instructions_of(program):
    return {name: p.signature.instructions for name, p in program.named_predictors()}


def demos_of(program):
    return {name: len(getattr(p, "demos", [])) for name, p in program.named_predictors()}


def main(optimizer_name="miprov2", auto="light", fake=False):
    setup_lm()
    retriever = Retriever(fake=fake, use_reranker=not fake)
    student = GSTQA(retriever)
    train, val = to_dspy("data/train.jsonl"), to_dspy("data/val.jsonl")
    before = instructions_of(student)

    if optimizer_name == "miprov2":
        opt = MIPROv2(metric=composite_metric, auto=auto, num_threads=1, seed=13)
        compiled = opt.compile(student, trainset=train, valset=val,
                               requires_permission_to_run=False)
    elif optimizer_name == "bootstrap":
        opt = BootstrapFewShotWithRandomSearch(
            metric=composite_metric, max_bootstrapped_demos=2,
            max_labeled_demos=2, num_candidate_programs=2, num_threads=1)
        compiled = opt.compile(student, trainset=train, valset=val)
    elif optimizer_name == "gepa":
        opt = dspy.GEPA(metric=gepa_metric, auto=auto, num_threads=1,
                        reflection_lm=dspy.LM(config.JUDGE_MODEL,
                                              api_key=config.OPENAI_API_KEY,
                                              temperature=1.0, max_tokens=4000))
        compiled = opt.compile(student, trainset=train, valset=val)
    else:
        raise ValueError(optimizer_name)

    Path(config.ARTIFACTS).mkdir(exist_ok=True)
    compiled.save(f"{config.ARTIFACTS}/optimized_program.json")

    after = instructions_of(compiled)
    lines = [f"# Prompt diff — {optimizer_name} (auto={auto})\n"]
    for name in before:
        lines += [f"\n## {name}\n", "### Before\n", "```", before[name], "```\n",
                  "### After\n", "```", after.get(name, ""), "```\n",
                  f"\nFew-shot demos attached: {demos_of(compiled).get(name, 0)}\n"]
    Path(f"{config.ARTIFACTS}/prompt_diff.md").write_text("\n".join(lines))
    print("saved artifacts/optimized_program.json and artifacts/prompt_diff.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", default="miprov2",
                    choices=["miprov2", "bootstrap", "gepa"])
    ap.add_argument("--auto", default="light", choices=["light", "medium", "heavy"])
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    main(a.optimizer, a.auto, a.fake)