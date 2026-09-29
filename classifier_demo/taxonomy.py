"""taxonomy YAML の読み込み。最低限のチェックのみ行う(網羅的なバリデータは作らない)。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import yaml

MAX_CHOICES = 51  # + none で最大52ラベル(A-Z・a-z。各ラベルは使用モデルで1トークン、1トークン目が互いに異なること)


@dataclass
class Choice:
    id: str
    name: str
    criteria: str
    lora_names: list[str] = field(default_factory=list)
    trigger_words: list[str] = field(default_factory=list)
    catch_all: bool = False


@dataclass
class Axis:
    id: str
    question: str
    multi: bool
    allow_none: bool
    choices: list[Choice]
    none_criteria: str | None = None


@dataclass
class Taxonomy:
    version: str
    axes: list[Axis]
    sha256: str

    def axis(self, axis_id: str) -> Axis:
        for a in self.axes:
            if a.id == axis_id:
                return a
        raise KeyError(f"unknown axis: {axis_id}")


def load(path: str) -> Taxonomy:
    with open(path, "rb") as f:
        raw = f.read()
    data = yaml.safe_load(raw)

    axes: list[Axis] = []
    seen_axis_ids: set[str] = set()
    for axis_data in data["axes"]:
        axis_id = axis_data["id"]
        if axis_id in seen_axis_ids:
            raise ValueError(f"duplicate axis id: {axis_id}")
        seen_axis_ids.add(axis_id)

        choices: list[Choice] = []
        seen_choice_ids: set[str] = set()
        for c in axis_data["choices"]:
            choice_id = c["id"]
            if choice_id in seen_choice_ids:
                raise ValueError(f"duplicate choice id in axis {axis_id}: {choice_id}")
            seen_choice_ids.add(choice_id)
            choices.append(
                Choice(
                    id=choice_id,
                    name=c["name"],
                    criteria=c["criteria"],
                    lora_names=list(c.get("lora_names", [])),
                    trigger_words=list(c.get("trigger_words", [])),
                    catch_all=bool(c.get("catch_all", False)),
                )
            )

        if len(choices) > MAX_CHOICES:
            raise ValueError(
                f"axis {axis_id} has {len(choices)} choices, max is {MAX_CHOICES} (+none)"
            )

        axes.append(
            Axis(
                id=axis_id,
                question=axis_data["question"],
                multi=bool(axis_data.get("multi", False)),
                allow_none=bool(axis_data.get("allow_none", False)),
                choices=choices,
                none_criteria=axis_data.get("none_criteria"),
            )
        )

    return Taxonomy(version=str(data["version"]), axes=axes, sha256=hashlib.sha256(raw).hexdigest())
