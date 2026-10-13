from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from firesight.benchmark import summarize_ms
from firesight.evaluation import Box, _write_reports, average_precision, box_iou, evaluate, evaluate_at_threshold, load_samples, match_class_predictions


class EvaluationMetricTests(unittest.TestCase):
    def test_iou_and_one_to_one_matching_counts_duplicates_as_false_positives(self):
        truth = Box(0, (0, 0, 10, 10))
        match = Box(0, (0, 0, 10, 10), .9)
        duplicate = Box(0, (0, 0, 10, 10), .8)
        self.assertEqual(box_iou(truth.xyxy, match.xyxy), 1.0)
        ranked, false_positives, false_negatives = match_class_predictions([match, duplicate], [truth], .5)
        self.assertEqual(ranked, [(.9, True), (.8, False)])
        self.assertEqual(len(false_positives), 1)
        self.assertEqual(false_negatives, [])

    def test_known_per_class_precision_recall_f1_and_map(self):
        truth = [(Box(0, (0, 0, 10, 10)), Box(2, (20, 20, 30, 30)))]
        predictions = [(Box(0, (0, 0, 10, 10), .9), Box(0, (40, 40, 50, 50), .8))]
        metrics = evaluate_at_threshold([(list(predictions[0]), list(truth[0]))], .5, .5)
        fire = metrics["by_class"]["fire"]
        smoke = metrics["by_class"]["smoke"]
        self.assertEqual((fire["tp"], fire["fp"], fire["fn"]), (1, 1, 0))
        self.assertAlmostEqual(fire["precision"], .5)
        self.assertEqual(fire["recall"], 1.0)
        self.assertAlmostEqual(fire["f1"], 2 / 3)
        self.assertEqual((smoke["tp"], smoke["fp"], smoke["fn"]), (0, 0, 1))
        self.assertEqual(smoke["recall"], 0)
        self.assertAlmostEqual(metrics["map50"], 0.5)
        self.assertAlmostEqual(average_precision([(.9, True), (.8, False)], 1), 1.0)

    def test_dataset_layout_fallback_records_yaml_path_issue_and_reads_empty_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_dir, label_dir = root / "valid" / "images", root / "valid" / "labels"
            image_dir.mkdir(parents=True); label_dir.mkdir(parents=True)
            image_path = image_dir / "empty.jpg"
            self.assertTrue(cv2.imwrite(str(image_path), np.zeros((12, 16, 3), dtype=np.uint8)))
            (label_dir / "empty.txt").write_text("", encoding="utf-8")
            data = root / "data.yaml"
            data.write_text("train: ../train/images\nval: ../valid/images\nnc: 3\nnames: [fire, person, smoke]\n", encoding="utf-8")
            samples, issues, dataset_id = load_samples(data, "val")
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].ground_truth, ())
            self.assertTrue(any("did not resolve" in issue for issue in issues))
            self.assertTrue(dataset_id.endswith("#val"))

    def test_report_generation_writes_json_csv_and_split_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            metrics = evaluate_at_threshold([], .4)
            metrics["macro_f1"] = 0.0
            metrics["macro_recall"] = 0.0
            report = {"split": "val", "dataset_samples": 0, "review_examples": []}
            json_path, csv_path = _write_reports(Path(temp), report, [metrics], min_alarm_recall=0)
            saved = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["validation_candidate"], .4)
            self.assertTrue(csv_path.is_file())
            self.assertIn("class_name", csv_path.read_text(encoding="utf-8-sig"))

    def test_final_test_split_rejects_threshold_selection(self):
        with self.assertRaisesRegex(ValueError, "not allowed on the final test split"):
            evaluate(data=Path("unused.yaml"), model_path=Path("unused.pt"), output=Path("unused-output"),
                     split="test", thresholds=[.2, .4])


class BenchmarkHelperTests(unittest.TestCase):
    def test_latency_summary_is_empty_safe_and_reports_percentiles(self):
        self.assertIsNone(summarize_ms([])["mean"])
        summary = summarize_ms([1, 2, 3, 4, 5])
        self.assertEqual(summary["mean"], 3)
        self.assertEqual(summary["median"], 3)
        self.assertGreaterEqual(summary["p95"], 4)


if __name__ == "__main__":
    unittest.main()
