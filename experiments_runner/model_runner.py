import typing as tp


class ModelRunner:
    def __init__(self, name: str):
        self.name = name

    def run(self, true_input: tp.Any) -> tp.Any:
        pass
