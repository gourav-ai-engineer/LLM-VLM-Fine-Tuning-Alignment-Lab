"""Framework-agnostic metrics used by experiment reports."""


def parameter_efficiency(trainable_parameters: int, total_parameters: int) -> float:
    if total_parameters <= 0:
        raise ValueError("total_parameters must be positive")
    if trainable_parameters < 0 or trainable_parameters > total_parameters:
        raise ValueError("trainable_parameters must be within total_parameters")
    return 100.0 * trainable_parameters / total_parameters


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)
