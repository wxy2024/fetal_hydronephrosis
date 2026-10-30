import unittest
import numpy as np
from PIL import Image
from scripts.evaluate_rcc_perturbation import perturb, discover_checkpoint
from pathlib import Path
import tempfile
class PerturbationTests(unittest.TestCase):
 def test_noise(self):
  im=Image.fromarray(np.full((32,32,3),128,dtype=np.uint8))
  a=np.asarray(perturb(im,'noise',.02,42));b=np.asarray(perturb(im,'noise',.02,42))
  np.testing.assert_array_equal(a,b)
  self.assertGreater(np.std(np.asarray(perturb(im,'noise',.1,42))),np.std(a))
  self.assertEqual(np.asarray(im).mean(),128)
 def test_checkpoint_missing_or_ambiguous(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   with self.assertRaises(FileNotFoundError):discover_checkpoint(p,'fold_1')
   (p/'fold_1').mkdir();(p/'fold_1/a.pt').touch()
   self.assertEqual(discover_checkpoint(p,'fold_1').name,'a.pt')
   (p/'fold_1/b.pt').touch()
   with self.assertRaises(ValueError):discover_checkpoint(p,'fold_1')
if __name__=='__main__':unittest.main()
