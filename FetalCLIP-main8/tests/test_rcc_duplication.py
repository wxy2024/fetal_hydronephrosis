import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from models.image_pooling_modules import AttentionImagePooling, ReliabilityCalibratedConsensusPooling
from models.longitudinal_gru_classifier import ModularGRUSideClassifier
from scripts.evaluate_rcc_duplication import (
    add_copies, effective_weights, measure_effect, aggregate_patients,
    bootstrap_summary, validate_selection, evaluate_sample, check_replay, save_plot,
)


class DuplicationTests(unittest.TestCase):
    def test_full_history_inference_with_missing_image_visit(self):
        import json
        torch.manual_seed(11)
        sample = dict(visits=[
            dict(visit_id='20200101', image_features=None, image_paths=[],
                 text_feature=torch.randn(8), num_images=0),
            dict(visit_id='20200201', image_features=torch.randn(3, 8),
                 image_paths=['a.png', 'b.png', 'c.png'], text_feature=torch.randn(8), num_images=3),
        ])
        for pooling in ['rcc_pooling', 'attention']:
            model = ModularGRUSideClassifier(feature_dim=8, image_pooling_type=pooling,
                visit_fusion_type='direct_concat', attn_hidden_dim=16,
                gru_hidden_dim=8, temporal_backbone='td_conv_gru').eval()
            base = evaluate_sample(model, sample, 1)
            after = evaluate_sample(model, add_copies(sample, 1, 0, 5), 1)
            self.assertEqual(len(after['beta']), 8)
            self.assertEqual(len(after['all_attention']), 2)
            self.assertTrue(0 <= after['probability'] <= 1)
            saved = dict(sample_id='synthetic', prob=base['probability'],
                         image_attention_json=json.dumps(base['all_attention']))
            self.assertEqual(check_replay(base, saved), 0)
            saved['prob'] = base['probability'] + .01
            with self.assertRaises(ValueError):
                check_replay(base, saved)

    def test_effective_weights_reconstruct_rcc_output(self):
        torch.manual_seed(3)
        pool = ReliabilityCalibratedConsensusPooling(8, 16).eval()
        x = torch.randn(3, 8)
        pooled, _, alpha = pool(x)
        beta, ratio = effective_weights(pool, alpha)
        torch.testing.assert_close(pooled, (beta[:, None] * x).sum(0))
        self.assertAlmostEqual(ratio, .5)
        self.assertAlmostEqual(float(beta.sum()), 1, places=6)

    def test_copies_preserve_original_and_total_mass(self):
        torch.manual_seed(7)
        pool = AttentionImagePooling(8, 16).eval()
        visit = dict(image_features=torch.randn(3, 8), image_paths=['a', 'b', 'c'], num_images=3)
        sample = dict(visits=[visit, dict(visit)], num_visits=2)
        modified = add_copies(sample, 0, 1, 3)
        self.assertEqual(len(sample['visits'][0]['image_paths']), 3)
        self.assertEqual(len(modified['visits'][0]['image_paths']), 6)
        self.assertIs(modified['visits'][1], sample['visits'][1])
        torch.testing.assert_close(modified['visits'][0]['image_features'][3:], visit['image_features'][1:2].repeat(3, 1))
        before, _, a = pool(visit['image_features'])
        after, _, b = pool(modified['visits'][0]['image_features'])
        effect = measure_effect(before, after, a, b, 1, 3, .2, .21)
        expected = 4 * float(a[1]) / (1 + 3 * float(a[1]))
        self.assertAlmostEqual(effect['duplicate_group_weight'], expected, places=6)
        self.assertAlmostEqual(effect['abs_probability_change'], .01)

    def test_identical_features_do_not_change_representation(self):
        torch.manual_seed(9)
        pool = ReliabilityCalibratedConsensusPooling(8, 16).eval()
        x = torch.randn(1, 8).repeat(3, 1)
        before, _, a = pool(x)
        after, _, b = pool(torch.cat([x, x[:1].repeat(5, 1)]))
        ea, _ = effective_weights(pool, a)
        eb, _ = effective_weights(pool, b)
        effect = measure_effect(before, after, ea, eb, 0, 5, .4, .4)
        self.assertLess(effect['relative_feature_change'], 1e-6)
        self.assertAlmostEqual(effect['duplicate_group_weight'], 6 / 8, places=6)

    def test_patient_weighting_and_paired_interval(self):
        rows = []
        for model in ['rcc_pooling', 'attention']:
            for case, n, value in [('a', 1, 1.), ('b', 9, 0.)]:
                for i in range(n):
                    rows.append(dict(model=model, case_id=case, fold_id='fold_1',
                        sample_id=case+'_left', image_index=i, added_copies=5,
                        group_weight_change=value, relative_feature_change=value,
                        cosine_distance=value, abs_probability_change=value))
        patients = aggregate_patients(rows)
        summary, paired = bootstrap_summary(patients, 100, 42)
        self.assertEqual(len(patients), 4)
        self.assertTrue(all(r['mean'] == .5 for r in summary))
        self.assertTrue(all(r['difference_rcc_minus_attention'] == 0 and r['ci_low'] == 0 and r['ci_high'] == 0 for r in paired))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'plot.png'
            save_plot(summary, path)
            self.assertGreater(path.stat().st_size, 0)

    def test_selection_rejects_one_image_and_duplicate_patient(self):
        import json
        pred = {'a_left': dict(case_id='a', sample_id='a_left', fold_id='fold_1',
            side='left', num_visits='1', visit_ids_used='v',
            image_attention_json=json.dumps([dict(visit_id='v', image_names=['one.png'])]))}
        selection = [dict(case_id='a', sample_id='a_left', fold_id='fold_1', visit_index='0', visit_id='v')]
        with self.assertRaises(ValueError):
            validate_selection(selection, pred)
        pred['a_left']['image_attention_json'] = json.dumps([dict(visit_id='v', image_names=['one.png', 'two.png'])])
        validate_selection(selection, pred)
        with self.assertRaises(ValueError):
            validate_selection(selection * 2, pred)


if __name__ == '__main__':
    unittest.main()
