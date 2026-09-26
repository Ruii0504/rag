import unittest
import logging
from run_regression import Observer, ModelCallRecorder, coverage


class RunnerTests(unittest.TestCase):
    def test_model_diagnostics_are_attached_to_current_case_only(self):
        observer = Observer()
        observer.record = {"calls": []}
        recorder = ModelCallRecorder(observer)
        recorder.emit(logging.LogRecord("models", logging.INFO, "", 0, '{"stage":"rerank","attempt":2}', (), None))
        self.assertEqual(observer.record["model_calls"], [{"stage": "rerank", "attempt": 2}])
        first = observer.record
        observer.record = {"calls": []}
        recorder.emit(logging.LogRecord("models", logging.INFO, "", 0, '{"stage":"chat"}', (), None))
        self.assertEqual(len(first["model_calls"]), 1)
        self.assertEqual(observer.record["model_calls"], [{"stage": "chat"}])

    def test_alternatives_and_multiple_groups(self):
        row = {"expected_behavior": "answer", "groundtruthchunkid": ["a", "b", "c"],
               "evidence_groups": [{"any_of": [{"chunk_ids": ["a"]}, {"chunk_ids": ["b"]}]},
                                   {"any_of": [{"chunk_ids": ["c"]}]}]}
        self.assertEqual(coverage(row, {"b"})["recall"], 0.5)
        self.assertTrue(coverage(row, {"b", "c"})["complete"])
        self.assertFalse(coverage(row, {"x"})["hit"])

    def test_refusal_excluded(self):
        self.assertIsNone(coverage({"expected_behavior": "refuse", "evidence_groups": []}, set()))

    def test_observer_preserves_return_and_records(self):
        class Target:
            def operation(self, value):
                return value + 1
        target = Target()
        observer = Observer()
        observer.record = {"calls": []}
        observer.wrap(target, "operation", "test")
        self.assertEqual(target.operation(4), 5)
        self.assertEqual(observer.record["calls"][0]["output"], 5)

    def test_observer_preserves_failure(self):
        class Target:
            def operation(self):
                raise ValueError("sensitive detail")
        observer = Observer()
        observer.record = {"calls": []}
        target = Target()
        observer.wrap(target, "operation", "test")
        with self.assertRaises(ValueError):
            target.operation()
        self.assertEqual(observer.record["calls"][0]["error_type"], "ValueError")
        self.assertNotIn("sensitive detail", str(observer.record))


if __name__ == "__main__":
    unittest.main()
