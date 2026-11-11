import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from PIL import Image

from evaluate import LABELS, metrics, read_manifest, run_evaluation
from prepare_dataset import prepare


def response(text='{"label":"happy"}', status="completed", refusal=False):
    return SimpleNamespace(id="test-response", model="gpt-6.1-luna", status=status,
                           output_text=text, usage=None,
                           output=[SimpleNamespace(content=[SimpleNamespace(type="refusal")])] if refusal else [])


class EvaluationTests(unittest.TestCase):
    def test_reproducible_balanced_sampling_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label in LABELS:
                (root / label).mkdir()
                for index in range(3):
                    Image.new("L", (48, 48), index * 30).save(root / label / f"{index}.png")
            first, second = root / "first.csv", root / "second.csv"
            prepare(root, first, per_class=2, seed=42)
            prepare(root, second, per_class=2, seed=42)
            self.assertEqual(first.read_text(), second.read_text())
            rows = read_manifest(first)
            self.assertEqual(len(rows), 14)
            self.assertEqual({label: sum(r["label"] == label for r in rows) for label in LABELS}, dict.fromkeys(LABELS, 2))
            with self.assertRaises(ValueError):
                prepare(root, first)

    def test_errors_unknown_and_class_imbalance_in_metrics(self):
        rows = [
            {"label": "happy", "prediction": "happy", "status": "ok"},
            {"label": "happy", "prediction": "unknown", "status": "ok"},
            {"label": "sad", "prediction": "happy", "status": "ok"},
            {"label": "fear", "prediction": None, "status": "error"},
        ]
        result = metrics(rows, 5)
        self.assertEqual(result["accuracy_including_errors"], 0.25)
        self.assertAlmostEqual(result["accuracy_valid_responses_only"], 1 / 3)
        self.assertAlmostEqual(result["macro_f1_supported_classes"], 1 / 6)
        self.assertAlmostEqual(result["balanced_accuracy_supported_classes"], 1 / 6)
        self.assertEqual(result["confusion_matrix"]["fear"]["error"], 1)
        self.assertFalse(result["run_complete"])

    def make_rows(self, root, count):
        path = root / "never-send-this-filename.png"
        Image.new("L", (48, 48), 128).save(path)
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["path", "label"])
            writer.writerow([path.name, "happy"])
        return read_manifest(manifest) * count

    def test_success_refusal_incomplete_and_invalid_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.make_rows(root, 4)
            client = Mock()
            client.responses.create.side_effect = [response(), response("", refusal=True),
                                                   response("", status="incomplete"), response("not-json")]
            result = run_evaluation(client, rows, root, "gpt-6.1-luna", "high")
            self.assertEqual(result["errors"], 3)
            self.assertEqual(result["accuracy_including_errors"], 0.25)
            self.assertTrue(result["run_complete"])
            request = client.responses.create.call_args_list[0].kwargs
            self.assertEqual(request["model"], "gpt-6.1-luna")
            self.assertFalse(request["store"])
            self.assertNotIn("never-send-this-filename", json.dumps(request))
            self.assertNotIn(str(root), json.dumps(request))
            saved = [json.loads(line) for line in (root / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(saved[1]["error"], "refusal")
            self.assertEqual(saved[2]["error"], "incomplete_response")

    def test_model_access_error_stops_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.make_rows(root, 2)
            client = Mock()
            error = RuntimeError("secret must not enter the log")
            error.status_code = 404
            client.responses.create.side_effect = error
            result = run_evaluation(client, rows, root, "gpt-6.1-luna", "high")
            self.assertEqual(client.responses.create.call_count, 1)
            self.assertFalse(result["run_complete"])
            self.assertEqual(result["attempted"], 1)
            self.assertEqual(result["accuracy_including_errors"], 0)
            self.assertNotIn("secret", (root / "predictions.jsonl").read_text())

    def test_empty_and_duplicate_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "manifest.csv"
            path.write_text("path,label\n")
            with self.assertRaises(ValueError):
                read_manifest(path)
            path.write_text("path,label\nx.png,happy\nx.png,sad\n")
            with self.assertRaises(ValueError):
                read_manifest(path)


if __name__ == "__main__":
    unittest.main()
