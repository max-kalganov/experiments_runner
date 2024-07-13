import typing as tp


class MetricRunner:
    def __init__(self, name: str):
        self.name = name

    def calculate(self, true_output: tp.Any, pred_output: tp.Any) -> tp.Any:
        pass

    def calculate_aggregated(self, single_results: tp.List[tp.Any]) -> tp.Any:
        pass
