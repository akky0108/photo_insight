from __future__ import annotations

from typing import Any, Dict, List, Optional

from photo_insight.core.batch_framework.base_batch import BaseBatchProcessor


class DummyBatchProcessor(BaseBatchProcessor):
    runtime_param_names = ("max_images",)

    def __init__(
        self,
        data: List[Dict[str, Any]],
        *,
        batch_size: int = 4,
        fail_batch_indexes: Optional[set[int]] = None,
        memory_stop_after_batches: Optional[int] = None,
    ) -> None:
        self._source_data = list(data)
        self.batch_size = batch_size
        self.max_images = None
        self.fail_batch_indexes = fail_batch_indexes or set()
        self.memory_stop_after_batches = memory_stop_after_batches
        self.processed_batches = 0
        self.submitted_batches_seen: List[int] = []
        self.processed_items_seen: List[int] = []
        self.memory_threshold_exceeded = False

        super().__init__(config_path=None, max_workers=2)

    def load_data(self, *args, **kwargs) -> List[Dict[str, Any]]:
        return list(self._source_data)

    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.processed_batches += 1
        self.submitted_batches_seen.append(len(batch))
        self.processed_items_seen.extend([int(row["id"]) for row in batch])

        if self.processed_batches in self.fail_batch_indexes:
            raise RuntimeError(f"batch {self.processed_batches} failed")

        if self.memory_stop_after_batches is not None and self.processed_batches >= self.memory_stop_after_batches:
            self.memory_threshold_exceeded = True

        return [
            {
                "id": row["id"],
                "status": "success",
                "score": float(row["id"]),
            }
            for row in batch
        ]


def _make_data(n: int) -> List[Dict[str, Any]]:
    return [{"id": i} for i in range(1, n + 1)]


def test_max_images_limits_actual_processed_items() -> None:
    processor = DummyBatchProcessor(_make_data(10), batch_size=4)

    processor.execute(max_images=3)

    assert processor.final_summary["total"] == 3
    assert processor.final_summary["success"] == 3
    assert processor.final_summary["failure"] == 0
    assert processor.final_summary["applied_max_images"] == 3
    assert processor.final_summary["submitted_items"] == 3
    assert processor.final_summary["input_items"] == 10
    assert processor.final_summary["stop_reason"] == "max_images_limit"

    assert processor.processed_count == 3
    assert processor.processed_items_seen == [1, 2, 3]
    assert processor.submitted_batches_seen == [3]
    assert processor.completed_all_batches is False


def test_max_images_splits_last_batch_by_remaining_items() -> None:
    processor = DummyBatchProcessor(_make_data(10), batch_size=4)

    processor.execute(max_images=6)

    assert processor.final_summary["total"] == 6
    assert processor.final_summary["success"] == 6
    assert processor.final_summary["failure"] == 0
    assert processor.final_summary["applied_max_images"] == 6
    assert processor.final_summary["submitted_items"] == 6
    assert processor.final_summary["stop_reason"] == "max_images_limit"

    assert processor.processed_items_seen == [1, 2, 3, 4, 5, 6]
    assert processor.submitted_batches_seen == [4, 2]


def test_without_max_images_processes_all_items() -> None:
    processor = DummyBatchProcessor(_make_data(9), batch_size=4)

    processor.execute()

    assert processor.final_summary["total"] == 9
    assert processor.final_summary["success"] == 9
    assert processor.final_summary["failure"] == 0
    assert processor.final_summary["applied_max_images"] is None
    assert processor.final_summary["submitted_items"] == 9
    assert processor.final_summary["input_items"] == 9
    assert processor.final_summary["stop_reason"] is None

    assert processor.processed_count == 9
    assert processor.processed_items_seen == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert processor.submitted_batches_seen == [4, 4, 1]
    assert processor.completed_all_batches is True


def test_non_positive_max_images_is_treated_as_unlimited() -> None:
    processor = DummyBatchProcessor(_make_data(5), batch_size=2)

    processor.execute(max_images=0)

    assert processor.final_summary["total"] == 5
    assert processor.final_summary["success"] == 5
    assert processor.final_summary["failure"] == 0
    assert processor.final_summary["applied_max_images"] is None
    assert processor.final_summary["submitted_items"] == 5
    assert processor.final_summary["stop_reason"] is None

    assert processor.processed_items_seen == [1, 2, 3, 4, 5]


def test_invalid_max_images_is_treated_as_unlimited() -> None:
    processor = DummyBatchProcessor(_make_data(5), batch_size=2)

    processor.execute(max_images="abc")

    assert processor.final_summary["total"] == 5
    assert processor.final_summary["success"] == 5
    assert processor.final_summary["failure"] == 0
    assert processor.final_summary["applied_max_images"] is None
    assert processor.final_summary["submitted_items"] == 5
    assert processor.final_summary["stop_reason"] is None

    assert processor.processed_items_seen == [1, 2, 3, 4, 5]


def test_max_images_takes_priority_at_submit_time() -> None:
    processor = DummyBatchProcessor(_make_data(10), batch_size=4)

    processor.execute(max_images=5)

    assert processor.final_summary["total"] == 5
    assert processor.final_summary["success"] == 5
    assert processor.final_summary["failure"] == 0
    assert processor.final_summary["applied_max_images"] == 5
    assert processor.final_summary["submitted_items"] == 5
    assert processor.final_summary["stop_reason"] == "max_images_limit"

    assert processor.processed_items_seen == [1, 2, 3, 4, 5]
    assert processor.submitted_batches_seen == [4, 1]


def test_memory_threshold_stops_early_when_max_images_not_reached() -> None:
    processor = DummyBatchProcessor(
        _make_data(12),
        batch_size=4,
        memory_stop_after_batches=1,
    )

    processor.execute(max_images=10)

    assert processor.final_summary["total"] <= 10
    assert processor.final_summary["submitted_items"] <= 10
    assert processor.final_summary["applied_max_images"] == 10
    assert processor.final_summary["stop_reason"] == "memory_threshold"
    assert processor.completed_all_batches is False


def test_failure_results_set_stop_reason_to_exception() -> None:
    processor = DummyBatchProcessor(
        _make_data(8),
        batch_size=4,
        fail_batch_indexes={1},
    )

    processor.execute()

    assert processor.get_stop_reason() == "exception"
    assert processor.completed_all_batches is False
    assert processor.final_summary["total"] <= 8
