"""Profiler evidence cannot turn missing counters into measured observations."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from development.kernels.observations import _profile_run, read_nsight


class ObservationTest(unittest.TestCase):
    def read(self, values, required=("a", "b")):
        header = (
            '"ID","Process ID","Kernel Name",'
            '"Metric Name","Metric Unit","Metric Value"\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            path.write_text(header + "\n".join(values))
            return read_nsight(path, required)

    def test_each_launch_requires_all_requested_counters(self):
        complete = ['"0","1","kernel","a","%","1.5"', '"0","1","kernel","b","","2"']
        self.assertEqual(self.read(complete)["kind"], "measured")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.read([*complete, '"1","1","kernel","a","%","3"'])

    def test_unavailable_and_nonfinite_counters_are_rejected(self):
        for value in ("N/A", "", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.read([f'"0","1","kernel","a","%","{value}"'], ("a",))

    def test_cancellation_reaps_the_owned_process_group(self):
        process = MagicMock(pid=12345)
        process.communicate.side_effect = KeyboardInterrupt
        with (
            patch("subprocess.Popen", return_value=process) as launch,
            patch("os.killpg") as kill,
            self.assertRaises(KeyboardInterrupt),
        ):
            _profile_run(["ncu"])
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        self.assertEqual([call.args[0] for call in kill.call_args_list], [12345, 12345])
        process.wait.assert_called()
