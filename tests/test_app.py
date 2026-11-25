import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


class PredictionPageTests(unittest.TestCase):
    def setUp(self):
        app.app.config["TESTING"] = True
        self.client = app.app.test_client()

    def test_home_shows_project_and_model_metrics(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Inside a cell", response.data)
        self.assertIn(b"HELD-OUT EVALUATION", response.data)
        self.assertIn(b"ROC-AUC", response.data)
        self.assertIn(b"not be used for health decisions", response.data)
        self.assertIn(b"Benign example", response.data)
        self.assertIn(b"Malignant example", response.data)
        self.assertIn(b"Feature correlation", response.data)
        self.assertIn(b"Multiple model comparison", response.data)
        self.assertIn(b"SHAP", response.data)
        self.assertIn(b"Browse the measurements", response.data)
        self.assertIn(b"How the app is put together", response.data)
        self.assertIn(b"Export PDF report", response.data)
        self.assertIn(b"PROJECT OBJECTIVE", response.data)
        self.assertIn(b"Build and evaluate a machine-learning classifier", response.data)
        self.assertIn(b"style.css?v=20261113-ui4", response.data)
        self.assertEqual(response.data.count(b' role="tab" '), 8)
        self.assertEqual(response.data.count(b'role="tabpanel"'), 8)
        self.assertIn(b'id="panel-workflow" role="tabpanel"', response.data)
        self.assertIn(b'id="panel-analysis" role="tabpanel"', response.data)

    def test_interactive_assets_are_available(self):
        stylesheet = self.client.get("/static/style.css")
        script = self.client.get("/static/app.js")
        try:
            self.assertEqual(stylesheet.status_code, 200)
            self.assertIn(b"--purple: #7653a6", stylesheet.get_data())
            self.assertEqual(script.status_code, 200)
            script_data = script.get_data()
            self.assertIn(b"window.cellScopeExamples", self.client.get("/").data)
            self.assertIn(b"ArrowRight", script_data)
            self.assertIn(b"aria-selected", script_data)
        finally:
            stylesheet.close()
            script.close()

    def test_valid_measurements_return_a_class_estimate(self):
        response = self.client.post(
            "/",
            data={
                "radius_mean": "14.2",
                "texture_mean": "20.1",
                "perimeter_mean": "92.4",
                "smoothness_mean": "0.1",
                "compactness_mean": "0.12",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"MODEL CLASS ESTIMATE", response.data)
        self.assertIn(b"How the trees voted", response.data)
        self.assertIn(b"Vote share, not medical confidence", response.data)
        self.assertIn(b"SHAP explanation for this sample", response.data)
        self.assertIn(b"perimeter mean", response.data)
        self.assertIn(b"PREDICTION SCORE", response.data)
        self.assertIn(b"Malignant:", response.data)
        self.assertIn(b"Benign:", response.data)
        self.assertIn(b"not a calibrated medical confidence", response.data)

    def test_pdf_report_download(self):
        response = self.client.get("/report")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/pdf")
        self.assertTrue(response.data.startswith(b"%PDF"))
        self.assertIn("attachment", response.headers["Content-Disposition"])

    def test_out_of_dataset_values_show_extrapolation_warning(self):
        response = self.client.post(
            "/",
            data={
                "radius_mean": "100",
                "texture_mean": "20.1",
                "perimeter_mean": "92.4",
                "smoothness_mean": "0.1",
                "compactness_mean": "0.12",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Input outside dataset range", response.data)
        self.assertIn(b"radius mean", response.data)

    def test_missing_measurement_shows_validation_message(self):
        response = self.client.post("/", data={"radius_mean": "14.2"})

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Please enter a value for every feature.", response.data)

    def test_negative_measurement_shows_validation_message(self):
        response = self.client.post(
            "/",
            data={
                "radius_mean": "-1",
                "texture_mean": "20.1",
                "perimeter_mean": "92.4",
                "smoothness_mean": "0.1",
                "compactness_mean": "0.12",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Enter valid, non-negative numbers", response.data)

    def test_infinite_measurement_shows_validation_message(self):
        response = self.client.post(
            "/",
            data={
                "radius_mean": "inf",
                "texture_mean": "20.1",
                "perimeter_mean": "92.4",
                "smoothness_mean": "0.1",
                "compactness_mean": "0.12",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Enter valid, non-negative numbers", response.data)


if __name__ == "__main__":
    unittest.main()
