"""
Pipelines entrypoint for evaluation_rank.
Currently delegates to legacy batch_processor implementation for compatibility.
"""

from photo_insight.batch_processor.evaluation_rank.evaluation_rank_batch_processor import EvaluationRankBatchProcessor

__all__ = ["EvaluationRankBatchProcessor"]
