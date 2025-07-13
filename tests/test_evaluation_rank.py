import os
import tempfile
import shutil
import csv
import pytest
from evaluation_rank_batch_processor import EvaluationRankBatchProcessor

@pytest.fixture
def dummy_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("""
paths:
  evaluation_data_dir: "{eval_dir}"
  output_data_dir: "{out_dir}"
weights:
  face:
    sharpness: 0.5
    contrast: 0.3
    noise: 0.1
    local_sharpness: 0.4
    local_contrast: 0.3
  general:
    sharpness: 0.4
    contrast: 0.3
    noise: 0.2
    local_sharpness: 0.3
    local_contrast: 0.2
  extra:
    composition: 0.1
    position: 0.05
    framing: 0.05
    direction: 0.05
""".format(eval_dir=tmp_path, out_dir=tmp_path))
    return str(config_path)

@pytest.fixture
def sample_csv(tmp_path):
    csv_path = tmp_path / "evaluation_results_2025-06-22.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file_name", "face_detected", "face_sharpness_score", "face_contrast_score",
            "face_noise_score", "face_local_sharpness_score", "face_local_contrast_score",
            "composition_rule_based_score", "face_position_score",
            "framing_score", "face_direction_score", "group_id"
        ])
        writer.writeheader()
        writer.writerow({
            "file_name": "test.jpg",
            "face_detected": "TRUE",
            "face_sharpness_score": "80",  # ↑ UP
            "face_contrast_score": "30",
            "face_noise_score": "5",
            "face_local_sharpness_score": "50",  # ↑ UP
            "face_local_contrast_score": "30",
            "composition_rule_based_score": "10",
            "face_position_score": "5",
            "framing_score": "5",
            "face_direction_score": "5",
            "group_id": "A"
        })
    return str(csv_path)

def test_evaluation_pipeline(tmp_path, dummy_config, sample_csv):
    processor = EvaluationRankBatchProcessor(
        config_path=dummy_config,
        date="2025-06-22"
    )
    processor.execute()

    # 出力ファイル確認
    merged_path = os.path.join(tmp_path, "evaluation_ranking_2025-06-22.csv")
    assert os.path.exists(merged_path)

    with open(merged_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["accepted_flag"] in {"1", "2"}

def test_flag_logic(dummy_config):
    processor = EvaluationRankBatchProcessor(config_path=dummy_config)
    entry = {
        "face_detected": "TRUE",
        "face_sharpness_score": "60",
        "face_contrast_score": "30",
        "face_noise_score": "5",
        "overall_evaluation": "80"
    }
    processor.assign_acceptance_flag(entry)
    assert entry["accepted_flag"] == 1

def test_sorted_output_order(tmp_path, dummy_config):
    # 2件の評価データ（高スコア・低スコア）を作成
    csv_path = tmp_path / "evaluation_results_2025-06-22.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file_name", "face_detected", "face_sharpness_score", "face_contrast_score",
            "face_noise_score", "face_local_sharpness_score", "face_local_contrast_score",
            "composition_rule_based_score", "face_position_score",
            "framing_score", "face_direction_score", "group_id"
        ])
        writer.writeheader()
        writer.writerow({  # entry A（スコア高）
            "file_name": "high.jpg",
            "face_detected": "TRUE",
            "face_sharpness_score": "90",
            "face_contrast_score": "35",
            "face_noise_score": "2",
            "face_local_sharpness_score": "60",
            "face_local_contrast_score": "35",
            "composition_rule_based_score": "10",
            "face_position_score": "5",
            "framing_score": "5",
            "face_direction_score": "5",
            "group_id": "A"
        })
        writer.writerow({  # entry B（スコア低）
            "file_name": "low.jpg",
            "face_detected": "TRUE",
            "face_sharpness_score": "50",
            "face_contrast_score": "20",
            "face_noise_score": "10",
            "face_local_sharpness_score": "30",
            "face_local_contrast_score": "20",
            "composition_rule_based_score": "5",
            "face_position_score": "3",
            "framing_score": "3",
            "face_direction_score": "3",
            "group_id": "A"
        })

    # 実行
    processor = EvaluationRankBatchProcessor(
        config_path=dummy_config,
        date="2025-06-22"
    )
    processor.execute()

    # 結果確認
    merged_path = os.path.join(tmp_path, "evaluation_ranking_2025-06-22.csv")
    with open(merged_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 2
        scores = [float(row["overall_evaluation"]) for row in rows]
        assert scores == sorted(scores, reverse=True), "Rows are not sorted by overall_evaluation descending."


def test_overall_evaluation_face_mode(dummy_config):
    processor = EvaluationRankBatchProcessor(config_path=dummy_config)
    entry = {
        "face_detected": "TRUE",
        "face_sharpness_score": "100",
        "face_contrast_score": "50",
        "face_noise_score": "0",
        "face_local_sharpness_score": "80",
        "face_local_contrast_score": "60",
        "composition_rule_based_score": "10",
        "face_position_score": "5",
        "framing_score": "5",
        "face_direction_score": "5",
    }
    processor.calculate_overall_evaluation(entry)
    assert isinstance(entry["overall_evaluation"], float)
    assert entry["overall_evaluation"] > 0


def test_top_35_percent_flag(dummy_config):
    processor = EvaluationRankBatchProcessor(config_path=dummy_config)
    batch = []
    for i in range(10):
        batch.append({
            "file_name": f"file_{i}.jpg",
            "face_detected": "TRUE",
            "overall_evaluation": str(100 - i),  # 100, 99, ..., 91
            "group_id": "A"
        })
    processor.rank_and_flag_top_entries(batch)
    flagged = [entry for entry in batch if entry["flag"] == 1]
    assert len(flagged) == 3  # 10 * 0.35 = 3.5 → 切り捨て or max(1, int(n * 0.35))


def test_load_evaluation_data_missing_file(dummy_config):
    processor = EvaluationRankBatchProcessor(config_path=dummy_config)
    with pytest.raises(FileNotFoundError):
        processor.load_evaluation_data("non_existent.csv")
