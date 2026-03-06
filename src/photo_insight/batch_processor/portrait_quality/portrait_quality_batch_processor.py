"""
Backward-compatible import path.

Old:
  photo_insight.batch_processor.portrait_quality.portrait_quality_batch_processor.PortraitQualityBatchProcessor

New:
  photo_insight.pipelines.portrait_quality.PortraitQualityBatchProcessor
"""

from photo_insight.pipelines.portrait_quality import PortraitQualityBatchProcessor

__all__ = ["PortraitQualityBatchProcessor"]
