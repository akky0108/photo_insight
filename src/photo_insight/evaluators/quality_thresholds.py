# evaluators/quality_thresholds.py
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class QualityThresholds:
    """
    config/quality_thresholds.yaml をロードして、
    プロファイル（portrait, landscape など）別に閾値を参照するためのクラス。
    """

    def __init__(self, path: str = "config/quality_thresholds.yaml") -> None:
        self.path = Path(path)
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"Quality threshold config not found: {self.path}")
        with self.path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # ここは yaml の構造に合わせて調整：
        # 例）top-level がすでに "quality_thresholds" ならそのまま返す
        return raw.get("quality_thresholds", raw)

    def profile(self, name: str) -> Dict[str, Any]:
        """
        プロファイル（portrait, landscape, default など）単位で dict を返す。
        無ければ空 dict。
        """
        return self.data.get(name, {})

    def get(self, *keys: str, default: Optional[Any] = None) -> Any:
        """
        共通の階層アクセス用:
        get("common", "noise", "min_score", default=0.5)
        """
        d: Any = self.data
        for k in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(k)
            if d is None:
                return default
        return d
