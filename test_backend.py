"""
DefectIntel — Comprehensive Backend Unit Tests
Run with: python test_backend.py
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import backend


class FakeModel:
    """Deterministic mock model for fast testing — no real embeddings."""
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        vectors = []
        for text in texts:
            if "display" in text.lower() or "screen" in text.lower():
                vectors.append([1.0, 0.0, 0.0, 0.0])
            elif "charg" in text.lower() or "usb" in text.lower():
                vectors.append([0.0, 1.0, 0.0, 0.0])
            elif "yarn" in text.lower() or "fabric" in text.lower():
                vectors.append([0.0, 0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


SAMPLE_SOURCE = pd.DataFrame({
    "report_id": ["R1", "R2", "R3", "R4", "R5"],
    "batch_id": ["B1", "B1", "B2", "B2", "B2"],
    "defect_description": [
        "Display flickers on startup",
        "Screen image distorted",
        "USB port fails to charge",
        "Display blinks during operation",
        "Charging port loose",
    ],
    "category": ["Display", "Display", "Charging", "Display", "Charging"],
    "severity": ["Low", "Medium", "High", "Medium", "Low"],
})


class BackendTests(unittest.TestCase):

    def setUp(self):
        self.df, self.embeddings = backend.analyze_dataset(SAMPLE_SOURCE, FakeModel())

    # ── Core analysis ────────────────────────────────────────────────────────

    def test_analysis_produces_valid_clusters(self):
        self.assertEqual(len(self.df), 5)
        self.assertIn("ai_cluster", self.df.columns)
        self.assertEqual(self.embeddings.shape[0], 5)

    def test_group_profiles_cover_all_reports(self):
        profiles = backend.group_profiles(self.df, self.embeddings)
        self.assertEqual(sum(p["report_count"] for p in profiles), 5)
        for p in profiles:
            self.assertIn("affected_batches", p)
            self.assertIn("categories", p)

    # ── Validation ───────────────────────────────────────────────────────────

    def test_missing_required_column_raises(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            backend.validate_dataset(pd.DataFrame({"batch_id": ["B1"]}))

    def test_empty_descriptions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "No usable reports"):
            backend.validate_dataset(pd.DataFrame({
                "batch_id": ["B1"],
                "defect_description": ["  "],
            }))

    # ── Similarity search ────────────────────────────────────────────────────

    def test_search_returns_ranked_results(self):
        results = backend.search_defect("Display is blinking", self.df, self.embeddings, model=FakeModel())
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("similarity", results[0])
        self.assertIn("data_origin", results[0])

    def test_empty_query_raises(self):
        with self.assertRaises(ValueError):
            backend.search_defect("", self.df, self.embeddings, model=FakeModel())

    # ── Batch analytics ──────────────────────────────────────────────────────

    def test_no_batch_pair_returns_empty_alerts(self):
        _, alerts = backend.batch_analytics(self.df)
        self.assertEqual(alerts, [])

    def test_batch_comparison_triggers_alert_with_recommendation(self):
        _, alerts = backend.batch_analytics(
            self.df,
            previous_batch="B1",
            current_batch="B2",
            min_current_count=1,
            increase_threshold=50,
            high_increase_threshold=100,
        )
        self.assertGreater(len(alerts), 0)
        for al in alerts:
            self.assertIn("recommendation", al)
            self.assertIn("alert_id", al)
            self.assertIn(al["severity"], ("High", "Medium"))

    def test_zero_baseline_reports_none_percentage(self):
        batches = pd.DataFrame({
            "batch_id": ["B1", "B2", "B2"],
            "ai_cluster": [0, 0, 1],
        })
        _, alerts = backend.batch_analytics(
            batches, previous_batch="B1", current_batch="B2", min_current_count=1
        )
        new_group = next((a for a in alerts if a["cluster"] == 1), None)
        self.assertIsNotNone(new_group)
        self.assertIsNone(new_group["increase_percent"])

    # ── Cluster assignment (centroid-based) ──────────────────────────────────

    def test_assign_to_cluster_suggested(self):
        q_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        cid, sim, atype = backend.assign_to_cluster(q_emb, self.df, self.embeddings, min_similarity=0.0)
        self.assertIsNotNone(cid)
        self.assertEqual(atype, "suggested")
        self.assertGreaterEqual(sim, 0.0)

    def test_assign_to_cluster_potential_new_group(self):
        q_emb = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        _, _, atype = backend.assign_to_cluster(q_emb, self.df, self.embeddings, min_similarity=0.99)
        self.assertEqual(atype, "potential_new_group")

    # ── Evaluation metrics ───────────────────────────────────────────────────

    def test_clustering_evaluation_produces_ari_nmi(self):
        result = backend.clustering_evaluation(self.df)
        self.assertIsNotNone(result)
        self.assertIn("adjusted_rand_index", result)
        self.assertIn("normalized_mutual_info", result)

    def test_silhouette_score_when_valid(self):
        # Only valid when 2+ clusters with fewer clusters than samples
        sil = backend.compute_silhouette(self.df, self.embeddings)
        # May be None if only 1 cluster was discovered — test that it doesn't crash
        self.assertTrue(sil is None or isinstance(sil, float))

    def test_full_evaluation_contains_expected_keys(self):
        result = backend.compute_full_evaluation(self.df, self.embeddings)
        self.assertIn("n_clusters", result)
        self.assertIn("n_samples", result)
        self.assertIn("silhouette", result)

    # ── Legacy acknowledgement (backward compat) ─────────────────────────────

    def test_legacy_acknowledge_and_retrieve(self):
        with tempfile.TemporaryDirectory() as td:
            ack_path = Path(td) / "ack.sqlite3"
            with patch.object(backend, "ACKNOWLEDGEMENT_PATH", ack_path):
                backend.acknowledge_alert("B1|B2|0")
                self.assertIn("B1|B2|0", backend.get_acknowledged_alerts())
                backend.unacknowledge_alert("B1|B2|0")
                self.assertNotIn("B1|B2|0", backend.get_acknowledged_alerts())
                backend.acknowledge_alert("X")
                backend.reset_acknowledgements()
                self.assertEqual(len(backend.get_acknowledged_alerts()), 0)

    # ── 4-State alert status system ──────────────────────────────────────────

    def test_alert_status_lifecycle(self):
        with tempfile.TemporaryDirectory() as td:
            st_path = Path(td) / "status.sqlite3"
            with patch.object(backend, "ALERT_STATUS_PATH", st_path):
                backend.set_alert_status("A1", "new")
                self.assertEqual(backend.get_alert_statuses()["A1"]["status"], "new")
                backend.set_alert_status("A1", "acknowledged", notes="Seen by QC lead")
                self.assertEqual(backend.get_alert_statuses()["A1"]["status"], "acknowledged")
                self.assertEqual(backend.get_alert_statuses()["A1"]["notes"], "Seen by QC lead")
                backend.set_alert_status("A1", "resolved")
                self.assertEqual(backend.get_alert_statuses()["A1"]["status"], "resolved")
                backend.reset_alert_statuses()
                self.assertEqual(backend.get_alert_statuses(), {})

    def test_invalid_status_raises(self):
        with tempfile.TemporaryDirectory() as td:
            st_path = Path(td) / "status.sqlite3"
            with patch.object(backend, "ALERT_STATUS_PATH", st_path):
                with self.assertRaises(ValueError):
                    backend.set_alert_status("A1", "invalid_status")

    # ── Recommendation ───────────────────────────────────────────────────────

    def test_recommendation_display_high(self):
        rec = backend.generate_investigation_recommendation("Display", "screen flickers", "High")
        self.assertIn("flex cable", rec.lower())
        self.assertIn("immediate quality alert", rec.lower())

    def test_recommendation_seal_domain(self):
        """Sealing keyword should produce a food packaging recommendation."""
        rec = backend.generate_investigation_recommendation("Seal", "seal failure on pouch", "Medium")
        self.assertIsInstance(rec, str)
        self.assertGreater(len(rec), 10)
        # The word 'seal' appears in the matched text so seal branch fires
        self.assertIn("seal", rec.lower())

    def test_recommendation_fallback(self):
        rec = backend.generate_investigation_recommendation(None, "unknown anomaly xyz", "Medium")
        self.assertIn("inspection", rec.lower())

    # ── Custom distance threshold ─────────────────────────────────────────────

    def test_custom_threshold_accepted(self):
        df_tight, embs = backend.analyze_dataset(SAMPLE_SOURCE, FakeModel(), distance_threshold=0.1)
        self.assertIn("ai_cluster", df_tight.columns)
        self.assertEqual(len(df_tight), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
