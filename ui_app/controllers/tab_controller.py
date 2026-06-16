from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ui_app.shell import TabSpec


@dataclass(frozen=True)
class TabController:
    """Small controller wrapper around a tab builder function."""

    key: str
    label: str
    builder: Callable[[], None]
    panel_classes: str = "q-pa-md"
    tab_style: str = ""

    def spec(self) -> TabSpec:
        return TabSpec(
            key=self.key,
            label=self.label,
            builder=self.builder,
            panel_classes=self.panel_classes,
            tab_style=self.tab_style,
        )
