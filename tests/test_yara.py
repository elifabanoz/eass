import unittest

from eass.layers.hashrep import EICAR_PARTS
from eass.layers.yara_scan import compile_rules
from eass.pipeline import RULE_PATH


class YaraTests(unittest.TestCase):
    def test_exact_eicar_bytes_match(self):
        matches = compile_rules(RULE_PATH).match(data=b"".join(EICAR_PARTS))
        self.assertEqual([match.rule for match in matches], ["EICAR_Test_File"])

    def test_marker_alone_does_not_match(self):
        matches = compile_rules(RULE_PATH).match(data=EICAR_PARTS[1])
        self.assertEqual(matches, [])

    def test_process_injection_string_combination_is_medium_confidence(self):
        sample = b"MZ" + b"VirtualAllocEx WriteProcessMemory CreateRemoteThread"
        matches = compile_rules(RULE_PATH).match(data=sample)
        self.assertEqual([match.rule for match in matches], ["Injection_API_Strings"])
        self.assertEqual(matches[0].meta["confidence"], "medium")


if __name__ == "__main__":
    unittest.main()
