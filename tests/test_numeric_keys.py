"""Guard against silent omission of generated manuscript numbers."""
import importlib.util,tempfile,unittest
from pathlib import Path
MODULE=Path(__file__).resolve().parents[1]/'scripts/build_paper_assets.py'
spec=importlib.util.spec_from_file_location('paper_assets',MODULE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class NumericKeysTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for folder in ['sections','appendices','generated']:(self.root/folder).mkdir()
        (self.root/'iclr2027_conference.tex').write_text(r'\input{sections/results}')
    def test_real_uses_and_escaped_percent_are_checked(self):
        (self.root/'sections/results.tex').write_text(r'\result{a} % \result{comment_only}'+'\n'+r'95\% interval: \result{b}')
        (self.root/'generated/rows.tex').write_text(r'\result{a}')
        out=m.validate_numeric_keys(self.root,{'a':{},'b':{}})
        self.assertEqual(out,{'literal_references':3,'distinct_keys':2,'missing':[]})
    def test_typo_fails_with_location(self):
        (self.root/'sections/results.tex').write_text(r'\result{misspelled_key}')
        with self.assertRaisesRegex(ValueError,r'misspelled_key.*sections/results.tex:1'):
            m.validate_numeric_keys(self.root,{'valid_key':{}})
    def test_empty_scan_cannot_pass_silently(self):
        with self.assertRaisesRegex(ValueError,'No literal numerical references'):
            m.validate_numeric_keys(self.root,{})
if __name__=='__main__':unittest.main()
