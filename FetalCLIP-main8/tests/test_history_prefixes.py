import unittest
import numpy as np
from scripts.evaluate_history_prefixes import metric_vector, METRICS, prefix_sample, holm, select_cohort
class HistoryTests(unittest.TestCase):
 def test_metrics(self):
  y=np.array([0,1,0,1,1,0]);p=np.array([.1,.5,.5,.8,.7,.3])
  expected=[5/6,1,2/3,3/4,1,6/7,8.5/9,np.mean((p-y)**2),.15]
  np.testing.assert_allclose(metric_vector(y,p),expected)
 def test_prefix(self):
  s={'visits':[{'visit_id':str(i)} for i in range(8)],'num_visits':8}
  a=prefix_sample(s,3);self.assertEqual(a['num_visits'],3);self.assertEqual(len(s['visits']),8)
  with self.assertRaises(ValueError):prefix_sample(s,9)
 def test_holm(self):np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
 def test_cohort(self):
  rows=[]
  for c,n in [('a',6),('b',5)]:
   for side in ['left','right']:rows.append(dict(case_id=c,side=side,num_visits=str(n),fold_id='fold_1',sample_id=c+'_'+side,label='0'))
  self.assertEqual({r['case_id'] for r in select_cohort(rows,6)}, {'a'})
if __name__=='__main__':unittest.main()
