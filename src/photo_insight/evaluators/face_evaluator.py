from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from photo_insight.face_detectors.insightface_evaluator import InsightFaceDetector


class FaceDetectorProtocol(Protocol):
    """顔検出 backend が満たすべき最小インターフェース。"""

    def available(self) -> bool: ...

    def detect(self, image: np.ndarray) -> Dict[str, Any]: ...


class FaceEvaluator:
    """
    顔検出バックエンドの薄いラッパー。

    現時点では InsightFace backend のみをサポートする。
    detector 実装固有の設定を受け取り、対応する detector に渡す。
    将来的に backend が増える場合は、本クラスを factory 的な入口として
    継続利用できるように、最小限の interface を明示している。
    """

    def __init__(
        self,
        backend: str = "insightface",
        confidence_threshold: float = 0.5,
        *,
        gpu: bool = False,
        strict: bool = False,
        model_name: str = "buffalo_l",
        model_root: str = "/work/models/insightface",
        providers: Optional[List[str]] = None,
        det_size: Tuple[int, int] = (640, 640),
    ):
        """
        Parameters
        ----------
        backend : str
            顔検出バックエンド名。現状は "insightface" のみ対応。
        confidence_threshold : float
            顔検出の信頼度しきい値。
        gpu : bool
            True の場合は GPU 利用を意図するが、providers 未指定時のみ補助的に使う。
        strict : bool
            True の場合、依存未導入や初期化失敗時に例外を送出する。
        model_name : str
            InsightFace のモデル名。
        model_root : str
            InsightFace モデルのルートディレクトリ。
        providers : Optional[List[str]]
            onnxruntime execution providers。
        det_size : Tuple[int, int]
            FaceAnalysis.prepare() に渡す検出サイズ。
        """
        self.backend = str(backend).strip().lower()

        if self.backend == "insightface":
            self.detector: FaceDetectorProtocol = InsightFaceDetector(
                confidence_threshold=confidence_threshold,
                gpu=gpu,
                strict=strict,
                model_name=model_name,
                model_root=model_root,
                providers=providers,
                det_size=det_size,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def available(self) -> bool:
        """
        backend detector が利用可能かどうかを返す。

        Returns
        -------
        bool
            detector が利用可能なら True
        """
        available_fn = getattr(self.detector, "available", None)
        if available_fn is None:
            return False
        return bool(available_fn())

    def evaluate(self, image: np.ndarray) -> Dict[str, Any]:
        """
        顔検出を実行する。

        Parameters
        ----------
        image : np.ndarray
            入力画像

        Returns
        -------
        Dict[str, Any]
            detector の検出結果
        """
        return self.detector.detect(image)
