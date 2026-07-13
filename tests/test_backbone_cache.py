import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn

from models import retrieval_model


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        flattened = images.flatten(start_dim=1)
        return flattened[:, : retrieval_model.DINOV2_ARCHS["s"]] * self.scale


class FrozenBackboneCacheTest(unittest.TestCase):
    def make_model(self, cache_dir):
        backbone = FakeBackbone()
        with mock.patch.object(retrieval_model, "load_dinov2_with_retry", return_value=backbone):
            model = retrieval_model.DinoWrapper(
                dino_size="s",
                feat_dim=2,
                backbone_tuning="frozen",
                use_cache=True,
                cache_dir=cache_dir,
            )
        return model, backbone

    def test_cache_reuses_raw_backbone_features_but_not_projection_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model, backbone = self.make_model(temp_dir)
            images = torch.arange(2 * 3 * 14 * 14, dtype=torch.float32).reshape(2, 3, 14, 14)
            model.eval()

            first = model.forward_eval(images, "cpu")
            with mock.patch.object(torch, "load", wraps=torch.load) as load_mock:
                second = model.forward_eval(images, "cpu")
            self.assertTrue(torch.allclose(first, second))
            self.assertEqual(backbone.calls, 1)
            load_mock.assert_not_called()

            with torch.no_grad():
                model.fc.bias.add_(torch.tensor([1.0, -1.0]))
            projected_after_head_change = model.forward_eval(images, "cpu")

            self.assertFalse(torch.allclose(second, projected_after_head_change))
            self.assertEqual(backbone.calls, 1)
            self.assertEqual(model.cache_stats()["fully_cached_batches"], 2)
            self.assertEqual(model.cache_stats()["batches_with_misses"], 1)
            self.assertEqual(model.cache_stats()["memory_hit_samples"], 4)
            self.assertEqual(model.cache_stats()["disk_hit_samples"], 0)
            self.assertEqual(model.cache_stats()["written_samples"], 2)
            cache_paths = list(model.cache_dir.glob("*.pt"))
            self.assertEqual(len(cache_paths), 2)
            for cache_path in cache_paths:
                cached_feature = torch.load(cache_path, map_location="cpu", weights_only=True)
                self.assertEqual(
                    cached_feature.untyped_storage().nbytes(),
                    cached_feature.numel() * cached_feature.element_size(),
                )

    def test_cache_is_shared_across_model_instances(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            images = torch.arange(2 * 3 * 14 * 14, dtype=torch.float32).reshape(2, 3, 14, 14)
            first_model, first_backbone = self.make_model(temp_dir)
            first_model.eval()
            first_model.forward_eval(images, "cpu")
            self.assertEqual(first_backbone.calls, 1)

            second_model, second_backbone = self.make_model(temp_dir)
            second_model.eval()
            second_model.forward_eval(images, "cpu")

            self.assertEqual(second_backbone.calls, 0)
            self.assertEqual(second_model.cache_stats()["hit_samples"], 2)
            self.assertEqual(second_model.cache_stats()["memory_hit_samples"], 0)
            self.assertEqual(second_model.cache_stats()["disk_hit_samples"], 2)

    def test_training_uses_cache_and_backbone_stays_in_eval_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model, backbone = self.make_model(temp_dir)
            images = torch.arange(2 * 3 * 14 * 14, dtype=torch.float32).reshape(2, 3, 14, 14)

            model.train()
            first = model.forward_cached(images, "cpu")
            second = model.forward_cached(images, "cpu")
            second.sum().backward()

            self.assertTrue(model.training)
            self.assertFalse(backbone.training)
            self.assertEqual(backbone.calls, 1)
            self.assertTrue(torch.allclose(first, second))
            self.assertEqual(model.cache_stats()["batches_with_misses"], 1)
            self.assertEqual(model.cache_stats()["fully_cached_batches"], 1)
            self.assertFalse(any(parameter.requires_grad for parameter in backbone.parameters()))
            self.assertTrue(any(parameter.requires_grad for parameter in model.fc.parameters()))
            self.assertTrue(all(parameter.grad is None for parameter in backbone.parameters()))
            self.assertTrue(any(parameter.grad is not None for parameter in model.fc.parameters()))

    def test_per_sample_cache_reuses_embeddings_across_different_batches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model, backbone = self.make_model(temp_dir)
            images = torch.arange(3 * 3 * 14 * 14, dtype=torch.float32).reshape(3, 3, 14, 14)
            model.eval()

            model.forward_eval(images[:2], "cpu")
            model.forward_eval(images[1:], "cpu")

            self.assertEqual(backbone.calls, 2)
            self.assertEqual(model.cache_stats()["hit_samples"], 1)
            self.assertEqual(model.cache_stats()["miss_samples"], 3)
            self.assertEqual(len(list(model.cache_dir.glob("*.pt"))), 3)

    def test_cache_requires_fully_frozen_backbone(self):
        with mock.patch.object(retrieval_model, "load_dinov2_with_retry", return_value=FakeBackbone()):
            with self.assertRaisesRegex(ValueError, "requires backbone_tuning='frozen'"):
                retrieval_model.DinoWrapper(
                    dino_size="s",
                    feat_dim=2,
                    backbone_tuning="full",
                    use_cache=True,
                )

    def test_cache_rejects_partial_backbone_tuning(self):
        with mock.patch.object(retrieval_model, "load_dinov2_with_retry", return_value=FakeBackbone()):
            with self.assertRaisesRegex(ValueError, "requires backbone_tuning='frozen'"):
                retrieval_model.DinoWrapper(
                    dino_size="s",
                    feat_dim=2,
                    backbone_tuning="last_1_block",
                    use_cache=True,
                )


if __name__ == "__main__":
    unittest.main()
