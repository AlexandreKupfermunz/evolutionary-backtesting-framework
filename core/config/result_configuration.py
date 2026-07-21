from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResultConfiguration:

    walk_forward_type: str
    fitness_function_name: str
    train_size_name: str | None
    configuration_folder: Path

    @property
    def name(self) -> str:
        parts = [
            self.walk_forward_type,
            self.fitness_function_name,
        ]

        if self.train_size_name is not None:
            parts.append(self.train_size_name)

        return " — ".join(parts)