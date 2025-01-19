import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/real-data')))

import unittest
from unittest.mock import patch
import requests
from ws_trust_prediction import check_qos, evaluate_trustworthiness

class TestQoSEvaluation(unittest.TestCase):
    def setUp(self):
        # Example service list for testing
        self.services = [
            {"name": "Service A", "url": "http://example.com/serviceA"},
            {"name": "Service B", "url": "http://example.com/serviceB"}
        ]

    @patch('requests.get')
    def test_check_qos_response_time(self, mock_get):
        # Mock a successful service response
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b"Test content"
        
        with patch('time.time', side_effect=[1, 2]):  # Simulate 1 second response
            results = check_qos(self.services)
            self.assertEqual(results[0]["Response Time (ms)"], 1000)  # Assert 1 second = 1000ms
            self.assertEqual(results[0]["Availability"], 1)  # Assert availability is 1 (true)

    @patch('requests.get')
    def test_check_qos_unavailable_service(self, mock_get):
        # Mock a failed service response
        mock_get.side_effect = requests.exceptions.RequestException("Service Unreachable")
        results = check_qos(self.services)
        self.assertEqual(results[0]["Availability"], 0)  # Assert availability is 0
        self.assertIsNone(results[0]["Response Time (ms)"])  # Assert response time is None

    def test_fuzzy_trustworthiness(self):
        # Mock inputs for evaluate_trustworthiness
        row = {
            "Response Time": 500,
            "Availability": 95,
            "Throughput": 10,
            "Reliability": 90
        }
        result = evaluate_trustworthiness(row)
        self.assertGreater(result, 0)  # Ensure trustworthiness score is calculated
        self.assertLessEqual(result, 100)  # Ensure score is within 0-100 range

    def test_throughput_calculation(self):
        # Verify throughput calculation
        row = {
            "Response Time (ms)": 2000,  # 2 seconds
            "Content Size (bytes)": 10000  # 10 KB
        }
        throughput = (row["Content Size (bytes)"] / row["Response Time (ms)"]) * 1000  # Convert to KB/s
        self.assertEqual(round(throughput, 2), 5.0)  # Assert correct throughput

if __name__ == "__main__":
    unittest.main()
