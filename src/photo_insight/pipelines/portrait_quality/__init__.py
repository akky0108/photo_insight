"""
Pipelines entrypoint for portrait_quality.
Currently delegates to legacy batch_processor implementation for compatibility.
"""

from .portrait_quality_batch_processor import PortraitQualityBatchProcessor

__all__ = ["PortraitQualityBatchProcessor"]