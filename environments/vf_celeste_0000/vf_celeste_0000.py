from datasets import load_dataset
import verifiers as vf

from string import ascii_letters, ascii_uppercase


def load_environment(**kwargs) -> vf.Environment:
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.map(lambda x: {
        "question": f"{x['instruction']}{': ' if x['input'] else ''}{x['input']}",
        "answer": "",
        "task": "capitalization",
    })

    def reward(completion, answer, **kwargs):
        n_letters = len([c for c in completion if c in ascii_letters])
        if n_letters == 0:
            return 0.0
        return len([c for c in completion if c in ascii_uppercase]) / n_letters

    return vf.SingleTurnEnv(
        dataset=dataset,
        rubric=vf.Rubric(funcs=[reward], weights=[1.0]),
    )