"""Negative checks prevent malformed annotations from silently passing validation."""
import copy
import unittest

from validate_dataset import HERE, read_jsonl, validate


class AnnotationValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = read_jsonl(HERE / "regression-100.jsonl")
        cls.chunks = read_jsonl(HERE / "chunks.snapshot.jsonl")

    def test_valid_dataset(self):
        self.assertEqual(validate(self.cases, self.chunks)["case_count"], 100)

    def test_missing_case_rejected(self):
        with self.assertRaisesRegex(ValueError, "100 cases"):
            validate(self.cases[:-1], self.chunks)

    def test_wrong_quota_rejected(self):
        rows = copy.deepcopy(self.cases)
        rows[0]["tier"] = "edge"
        with self.assertRaisesRegex(ValueError, "tier quota"):
            validate(rows, self.chunks)

    def test_unknown_chunk_rejected(self):
        rows = copy.deepcopy(self.cases)
        rows[0]["evidence_groups"][0]["any_of"][0]["chunk_ids"] = ["fabricated:0"]
        with self.assertRaisesRegex(ValueError, "unknown chunk"):
            validate(rows, self.chunks)

    def test_fabricated_quote_rejected(self):
        rows = copy.deepcopy(self.cases)
        rows[0]["evidence_groups"][0]["any_of"][0]["quotes"][0]["text"] = "This quotation does not exist in the source."
        with self.assertRaisesRegex(ValueError, "quote not found"):
            validate(rows, self.chunks)

    def test_wrong_document_rejected(self):
        rows = copy.deepcopy(self.cases)
        rows[0]["evidence_groups"][0]["any_of"][0]["document_ids"] = ["wrong-document"]
        with self.assertRaisesRegex(ValueError, "document mismatch"):
            validate(rows, self.chunks)

    def test_snapshot_mutation_rejected(self):
        chunks = copy.deepcopy(self.chunks)
        chunks[0]["content"] += " changed"
        with self.assertRaisesRegex(ValueError, "Chunk hash mismatch"):
            validate(self.cases, chunks)


if __name__ == "__main__":
    unittest.main()
