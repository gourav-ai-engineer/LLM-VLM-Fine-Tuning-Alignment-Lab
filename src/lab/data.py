"""Lightweight dataset validation utilities independent of a training framework."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetReport:
    total_rows: int
    valid_rows: int
    duplicate_rows: int
    missing_prompt_rows: int
    missing_response_rows: int

    @property
    def validity_rate(self) -> float:
        return self.valid_rows / self.total_rows if self.total_rows else 0.0


def validate_instruction_rows(rows: Iterable[Mapping[str, object]]) -> DatasetReport:
    total = valid = duplicates = missing_prompt = missing_response = 0
    seen: set[tuple[str, str]] = set()

    for row in rows:
        total += 1
        prompt = str(row.get("prompt", "")).strip()
        response = str(row.get("response", "")).strip()
        if not prompt:
            missing_prompt += 1
        if not response:
            missing_response += 1
        key = (prompt, response)
        is_duplicate = key in seen
        if is_duplicate:
            duplicates += 1
        seen.add(key)
        if prompt and response and not is_duplicate:
            valid += 1

    return DatasetReport(total, valid, duplicates, missing_prompt, missing_response)
