from datasets import load_dataset
import verifiers as vf

from string import ascii_uppercase


def load_environment(**kwargs) -> vf.Environment:
    dataset = load_dataset("tatsu-lab/alpaca")
    dataset = dataset.map(lambda x: {
        "question": f"{x['instruction']}{': ' if x['input'] else ''}{x['input']}",
        "answer": {},
        "task": "capitalization",
    })

    def reward(completion, answer, **kwargs):
        return len([c for c in completion if c in ascii_uppercase]) / len(completion)

    return vf.Environment(
        dataset=dataset,
        rubric=vf.Rubric(funcs=[reward], weights=[1.0]),
    )