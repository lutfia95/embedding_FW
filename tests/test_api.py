import importlib.util
import os
import tempfile
import time
import unittest

HAS_API_DEPS = all(importlib.util.find_spec(name) is not None for name in ("fastapi", "httpx"))


@unittest.skipUnless(HAS_API_DEPS, "Install plms-framework[api] and httpx")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        os.environ["PLMS_ARTIFACT_DIR"] = cls.directory.name
        from fastapi.testclient import TestClient

        from plms_framework.api import app, jobs

        cls.client = TestClient(app)
        cls.jobs = jobs

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_registry_and_embedding(self):
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(response.json()["count"], 14)
        response = self.client.post(
            "/v1/embed",
            json={
                "models": ["reference"],
                "sequences": [{"id": "protein", "sequence": "MKT"}],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("reference-deterministic-v1", response.json()["models"])

    def test_async_comparison_and_artifacts(self):
        response = self.client.post(
            "/v1/compare",
            json={
                "models": ["reference"],
                "sequences": [
                    {"id": "a", "sequence": "MKTAYI"},
                    {"id": "b", "sequence": "MKTAVI"},
                ],
                "metrics": ["cosine_similarity", "sequence_identity"],
                "visualizations": ["clustered_heatmap"],
            },
        )
        self.assertEqual(response.status_code, 202)
        job_id = response.json()["id"]
        job = None
        for _ in range(100):
            job = self.client.get(f"/v1/jobs/{job_id}").json()
            if job["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)
        self.assertEqual(job["status"], "succeeded")
        artifacts = self.client.get(f"/v1/jobs/{job_id}/artifacts").json()["artifacts"]
        self.assertTrue(
            {"result.json", "manifest.json", "clustered_heatmap.svg"}.issubset(
                {a["name"] for a in artifacts}
            )
        )

    def test_fasta_upload(self):
        response = self.client.post(
            "/v1/fasta/parse",
            files={"file": ("proteins.fasta", b">a\nMKT\n>b\nAAA\n", "text/plain")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["id"] for item in response.json()["sequences"]], ["a", "b"])

    def test_cross_compare_reuses_completed_jobs(self):
        def source_result(model_id, vectors):
            return {
                "sequence_order": ["a", "b", "c"],
                "models": {
                    model_id: {
                        "manifest": {"id": model_id},
                        "representations": [
                            {
                                "sequence_id": sequence_id,
                                "method": "mean",
                                "layer": "model_default",
                                "values": vector,
                            }
                            for sequence_id, vector in vectors.items()
                        ],
                    }
                },
            }

        first = self.jobs.submit(
            "synthetic",
            lambda: source_result("model-a", {"a": [1, 0], "b": [0, 1], "c": [-1, 0]}),
        )
        second = self.jobs.submit(
            "synthetic",
            lambda: source_result("model-b", {"a": [0, 2, 0], "b": [-2, 0, 0], "c": [0, -2, 0]}),
        )
        for source in (first, second):
            for _ in range(100):
                if self.jobs.get(source.id).status in {"succeeded", "failed"}:
                    break
                time.sleep(0.01)
            self.assertEqual(self.jobs.get(source.id).status, "succeeded")
        response = self.client.post(
            "/v1/cross-compare",
            json={
                "sources": [{"job_id": first.id}, {"job_id": second.id}],
                "methods": ["matrix_spearman", "linear_cka", "consensus_rank"],
                "visualizations": ["model_agreement", "consensus_heatmap"],
            },
        )
        self.assertEqual(response.status_code, 202, response.text)
        job_id = response.json()["id"]
        for _ in range(100):
            job = self.client.get(f"/v1/jobs/{job_id}?include_result=true").json()
            if job["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)
        self.assertEqual(job["status"], "succeeded", job)
        self.assertTrue(job["result"]["cross_model"]["comparisons"]["linear_cka"]["available"])
        artifact_names = {
            item["name"]
            for item in self.client.get(f"/v1/jobs/{job_id}/artifacts").json()["artifacts"]
        }
        self.assertIn("model_agreement.svg", artifact_names)
        self.assertIn("consensus_heatmap.svg", artifact_names)

    def test_imported_spaces_can_be_cross_compared(self):
        source_jobs = []
        for name, values in (
            (
                "external-esm",
                {"a": [1, 0], "b": [0, 1], "c": [-1, 0]},
            ),
            (
                "external-t5",
                {"a": [2, 0, 0], "b": [0, 3, 0], "c": [-4, 0, 0]},
            ),
        ):
            response = self.client.post(
                "/v1/spaces/import",
                json={
                    "space_name": name,
                    "source": {
                        "provider": "test-suite",
                        "revision": "fixture-v1",
                    },
                    "representations": [
                        {
                            "method": "mean",
                            "normalization": "l2",
                            "embeddings": [
                                {"sequence_id": sequence_id, "values": vector}
                                for sequence_id, vector in values.items()
                            ],
                        }
                    ],
                },
            )
            self.assertEqual(response.status_code, 202, response.text)
            source_jobs.append(response.json()["id"])

        for job_id in source_jobs:
            for _ in range(100):
                job = self.client.get(f"/v1/jobs/{job_id}?include_result=true").json()
                if job["status"] != "running" and job["status"] != "queued":
                    break
                time.sleep(0.01)
            self.assertEqual(job["status"], "succeeded", job)
            imported_model = next(iter(job["result"]["models"].values()))
            self.assertEqual(
                imported_model["manifest"]["conformance"],
                "externally_supplied_structural_validation_only",
            )

        response = self.client.post(
            "/v1/cross-compare",
            json={
                "sources": [{"job_id": job_id} for job_id in source_jobs],
                "methods": ["matrix_spearman", "linear_cka", "neighbor_overlap"],
                "neighbor_k": [1],
            },
        )
        self.assertEqual(response.status_code, 202, response.text)
        cross_job_id = response.json()["id"]
        for _ in range(100):
            cross_job = self.client.get(f"/v1/jobs/{cross_job_id}?include_result=true").json()
            if cross_job["status"] != "running" and cross_job["status"] != "queued":
                break
            time.sleep(0.01)
        self.assertEqual(cross_job["status"], "succeeded", cross_job)
        result = cross_job["result"]["cross_model"]
        self.assertEqual(
            result["relationship_model_order"],
            ["external-esm", "external-t5"],
        )
        self.assertTrue(result["comparisons"]["matrix_spearman"]["available"])
        self.assertTrue(result["comparisons"]["linear_cka"]["available"])

    def test_import_rejects_coercible_non_numeric_values(self):
        response = self.client.post(
            "/v1/spaces/import",
            json={
                "space_name": "bad-import",
                "representations": [
                    {
                        "method": "mean",
                        "embeddings": [
                            {"sequence_id": "a", "values": [True, "1"]},
                        ],
                    }
                ],
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_diagnostics_reuse_import_job(self):
        response = self.client.post(
            "/v1/spaces/import",
            json={
                "space_name": "diagnostic-space",
                "representations": [
                    {
                        "method": "mean",
                        "embeddings": [
                            {"sequence_id": "a", "values": [1, 0]},
                            {"sequence_id": "b", "values": [1, 0]},
                            {"sequence_id": "c", "values": [0, 1]},
                        ],
                    }
                ],
            },
        )
        import_job_id = response.json()["id"]
        for _ in range(100):
            source = self.client.get(f"/v1/jobs/{import_job_id}?include_result=true").json()
            if source["status"] not in {"running", "queued"}:
                break
            time.sleep(0.01)
        self.assertEqual(source["status"], "succeeded", source)

        response = self.client.post(
            "/v1/diagnostics",
            json={"source_job_id": import_job_id, "hubness_k": 1},
        )
        self.assertEqual(response.status_code, 202, response.text)
        diagnostic_job_id = response.json()["id"]
        for _ in range(100):
            job = self.client.get(f"/v1/jobs/{diagnostic_job_id}?include_result=true").json()
            if job["status"] not in {"running", "queued"}:
                break
            time.sleep(0.01)
        self.assertEqual(job["status"], "succeeded", job)
        diagnostics = job["result"]["diagnostics"]
        self.assertTrue(diagnostics["available"])
        model = diagnostics["models"]["diagnostic-space"]
        self.assertEqual(model["duplicates"]["groups"], [["a", "b"]])
        self.assertTrue(model["hubness"]["available"])

    def test_mean_centering_creates_derived_job_and_parameter_artifact(self):
        response = self.client.post(
            "/v1/spaces/import",
            json={
                "space_name": "transform-source",
                "representations": [
                    {
                        "method": "mean",
                        "normalization": "none",
                        "embeddings": [
                            {"sequence_id": "a", "values": [1, 2]},
                            {"sequence_id": "b", "values": [3, 4]},
                            {"sequence_id": "c", "values": [5, 6]},
                        ],
                    }
                ],
            },
        )
        source_job_id = response.json()["id"]
        for _ in range(100):
            source = self.client.get(f"/v1/jobs/{source_job_id}").json()
            if source["status"] not in {"running", "queued"}:
                break
            time.sleep(0.01)
        self.assertEqual(source["status"], "succeeded", source)

        response = self.client.post(
            "/v1/transform",
            json={
                "source_job_id": source_job_id,
                "method": "mean_center",
                "fit_sequence_ids": ["a", "b"],
                "hubness_k": 1,
            },
        )
        self.assertEqual(response.status_code, 202, response.text)
        job_id = response.json()["id"]
        for _ in range(100):
            job = self.client.get(f"/v1/jobs/{job_id}?include_result=true").json()
            if job["status"] not in {"running", "queued"}:
                break
            time.sleep(0.01)
        self.assertEqual(job["status"], "succeeded", job)
        values = [
            item["values"]
            for item in job["result"]["models"]["transform-source::mean-centered"][
                "representations"
            ]
        ]
        self.assertEqual(values, [[-1.0, -1.0], [1.0, 1.0], [3.0, 3.0]])
        self.assertIn("before", job["result"]["diagnostics"])
        artifact_names = {
            item["name"]
            for item in self.client.get(f"/v1/jobs/{job_id}/artifacts").json()["artifacts"]
        }
        self.assertIn("transformation_parameters.json", artifact_names)

    def test_categorical_annotation_analysis(self):
        response = self.client.post(
            "/v1/spaces/import",
            json={
                "space_name": "annotation-space",
                "representations": [
                    {
                        "method": "mean",
                        "embeddings": [
                            {"sequence_id": "a1", "values": [1.0, 0.0]},
                            {"sequence_id": "a2", "values": [0.9, 0.1]},
                            {"sequence_id": "b1", "values": [-1.0, 0.0]},
                            {"sequence_id": "b2", "values": [-0.9, -0.1]},
                        ],
                    }
                ],
            },
        )
        source_job_id = response.json()["id"]
        for _ in range(100):
            source = self.client.get(f"/v1/jobs/{source_job_id}").json()
            if source["status"] not in {"running", "queued"}:
                break
            time.sleep(0.01)
        self.assertEqual(source["status"], "succeeded", source)

        response = self.client.post(
            "/v1/annotations/analyze",
            json={
                "source_job_id": source_job_id,
                "feature": "family",
                "k": [1],
                "annotations": [
                    {
                        "sequence_id": sequence_id,
                        "categorical": {"family": sequence_id[0].upper()},
                    }
                    for sequence_id in ("a1", "a2", "b1", "b2")
                ],
            },
        )
        self.assertEqual(response.status_code, 202, response.text)
        job_id = response.json()["id"]
        for _ in range(100):
            job = self.client.get(f"/v1/jobs/{job_id}?include_result=true").json()
            if job["status"] not in {"running", "queued"}:
                break
            time.sleep(0.01)
        self.assertEqual(job["status"], "succeeded", job)
        scores = job["result"]["annotation_analysis"]["models"]["annotation-space"]["by_k"]["1"]
        self.assertEqual(scores["micro_consistency"], 1.0)
        self.assertEqual(scores["class_mixing"]["counts"], [[2, 0], [0, 2]])


if __name__ == "__main__":
    unittest.main()
