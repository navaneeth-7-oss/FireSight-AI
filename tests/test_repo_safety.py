from __future__ import annotations

import unittest

from scripts.check_repo_safety import _text_findings


class RepositorySafetyScannerTests(unittest.TestCase):
    def test_credential_bearing_rtsp_url_is_reported_without_secret_value(self):
        value = "rtsp://" + "private-user:private-pass@" + "camera.example" + "/stream"
        findings = _text_findings("service.py", value)
        self.assertEqual(len(findings), 1)
        self.assertIn("credential-bearing", findings[0][1])
        self.assertNotIn("private-pass", str(findings))

    def test_private_rtsp_address_is_reported_even_without_userinfo(self):
        value = "rtsps://" + "192.168.1.10" + ":322/live"
        findings = _text_findings("service.py", value)
        self.assertEqual(len(findings), 1)
        self.assertIn("private-network", findings[0][1])

    def test_private_local_hostname_is_reported(self):
        value = "rtsp://" + "private-camera.local/live"
        findings = _text_findings("service.py", value)
        self.assertEqual(len(findings), 1)
        self.assertIn("private-network", findings[0][1])

    def test_fake_rtsp_test_fixture_is_allowed(self):
        value = "rtsp://" + "viewer:private@" + "camera.local/live?token=secret"
        self.assertEqual(_text_findings("tests/test_repo_safety.py", value), [])

    def test_high_entropy_secret_assignment_is_redacted(self):
        value = "API_" + "KEY=" + "AbCdEf0123456789_XyZ987"
        findings = _text_findings("settings.py", value)
        self.assertEqual(len(findings), 1)
        self.assertIn("redacted", findings[0][1])
        self.assertNotIn("AbCdEf", str(findings))


if __name__ == "__main__":
    unittest.main()
