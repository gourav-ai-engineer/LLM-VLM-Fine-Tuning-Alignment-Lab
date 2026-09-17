from lab.metrics.parameter_counts import parameter_counts, trainable_percentage


class FakeParameter:
    def __init__(self, count: int, requires_grad: bool):
        self._count = count
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._count


def test_parameter_counts():
    total, trainable = parameter_counts([
        FakeParameter(10, True),
        FakeParameter(20, False),
    ])
    assert (total, trainable) == (30, 10)


def test_trainable_percentage():
    assert trainable_percentage(200, 50) == 25.0
