"""
Pipelines entrypoint for portrait_quality.
Currently delegates to legacy batch_processor implementation for compatibility.
"""

from photo_insight.batch_processor.portrait_quality import PortraitQualityBatchProcessor

__all__ = ["PortraitQualityBatchProcessor"]
