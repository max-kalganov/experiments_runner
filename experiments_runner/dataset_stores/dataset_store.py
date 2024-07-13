import typing as tp


class DatasetStore:
    def __init__(self, name: str):
        self.name = name

    def yield_samples(self) -> tp.Iterable[tp.Tuple[tp.Any, tp.Any]]:
        pass
