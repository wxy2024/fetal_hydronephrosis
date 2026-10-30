import unittest
import tempfile
from pathlib import Path
import torch
from utils.checkpoint_io import save_final_checkpoint
class CheckpointTests(unittest.TestCase):
 def test_roundtrip_and_no_overwrite(self):
  model=torch.nn.Linear(3,1).eval();x=torch.randn(2,3);before=model(x).detach()
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'fold_1/final_model.pt'
   save_final_checkpoint(model,path,{'fold_id':'fold_1','model_kwargs':{'input_dim':3},'train_case_ids':['a'],'test_case_ids':['b']})
   saved=torch.load(path,map_location='cpu',weights_only=True)
   clone=torch.nn.Linear(3,1);clone.load_state_dict(saved['model_state_dict'])
   torch.testing.assert_close(clone(x),before)
   self.assertEqual(saved['metadata']['test_case_ids'],['b'])
   self.assertTrue(path.with_suffix('.json').is_file())
   with self.assertRaises(FileExistsError):save_final_checkpoint(model,path,{})
 def test_reject_leakage(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):save_final_checkpoint(torch.nn.Linear(1,1),Path(d)/'final_model.pt',{'train_case_ids':['a'],'test_case_ids':['a']})
if __name__=='__main__':unittest.main()
