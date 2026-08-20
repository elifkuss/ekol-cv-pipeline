import unittest
import numpy as np
from src.preprocessor import DataScrubber
from src.state_machine import WarehouseStateMachine
from src.pipeline import PerformanceTracker

class TestWarehouseStateMachine(unittest.TestCase):
    def setUp(self):
        self.state_machine = WarehouseStateMachine()

    def test_set_active_order(self):
        self.state_machine.set_active_order("GOMLEK")
        self.assertEqual(self.state_machine.active_order, "GOMLEK")

    def test_check_packaging_zone_in_transit(self):
        self.state_machine.set_active_order("GOMLEK")
        xyxy = np.array([2])  # Left area of screen (In Transit)
        status = self.state_machine.check_packaging_zone(1, "tie", xyxy, (480, 640))
        self.assertEqual(status, "IN_TRANSIT")

    def test_check_packaging_zone_success(self):
        self.state_machine.set_active_order("GOMLEK")
        xyxy = np.array([2])  # Right area of screen (Packaging Zone)
        status = self.state_machine.check_packaging_zone(5, "tie", xyxy, (480, 640))
        self.assertEqual(status, "SUCCESS")

    def test_check_packaging_zone_anomaly(self):
        self.state_machine.set_active_order("GOMLEK")
        xyxy = np.array([2])  # Right area of screen (Packaging Zone)
        status = self.state_machine.check_packaging_zone(4, "backpack", xyxy, (480, 640))
        self.assertEqual(status, "ANOMALY_MISMATCH")

    def test_already_processed(self):
        self.state_machine.set_active_order("GOMLEK")
        xyxy = np.array([2])
        self.state_machine.check_packaging_zone(3, "tie", xyxy, (480, 640))
        status2 = self.state_machine.check_packaging_zone(3, "tie", xyxy, (480, 640))
        self.assertEqual(status2, "ALREADY_PROCESSED")


class TestDataScrubber(unittest.TestCase):
    def test_anonymize_workers(self):
        scrubber = DataScrubber()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        class MockBox:
            def __init__(self):
                self.cls = [np.array()] # Person class ID
                self.xyxy = [np.array([2])]

        class MockResults:
            def __init__(self):
                self.boxes = [MockBox()]

        processed = scrubber.anonymize_workers(frame, MockResults())
        # Check if the face area pixels have changed (blur applied)
        self.assertFalse(np.array_equal(frame[10:27, 10:80], np.zeros((17, 70, 3))))


class TestPerformanceTracker(unittest.TestCase):
    def test_performance_tracking(self):
        tracker = PerformanceTracker()
        tracker.record_frame(20.0)
        tracker.record_frame(30.0)
        tracker.record_event("SUCCESS")
        
        report = tracker.generate_report()
        self.assertEqual(report["successful_packagings"], 1)
        self.assertEqual(report["average_latency_ms"], 25.0)


if __name__ == "__main__":
    print("\n--- [EKOL CV PIPELINE] RUNNING UNIT TESTS ---\n")
    unittest.main()
