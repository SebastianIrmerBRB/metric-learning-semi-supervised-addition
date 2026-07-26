import gc
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset

from models import retrieval_model
from training import engine
from training.frozen_feature_cache import FrozenFeatureDatasetCache


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        flattened = images.flatten(start_dim=1)
        return flattened[:, : retrieval_model.DINOV2_ARCHS["s"]] * self.scale


class StableIdentityTransform:
    def __call__(self, image):
        return image

    def __repr__(self):
        return "StableIdentityTransform()"


class IndexedImageDataset(Dataset):
    access_log = []

    def __init__(self):
        self.images = torch.arange(
            4 * 3 * 14 * 14,
            dtype=torch.float32,
        ).reshape(4, 3, 14, 14)
        self.orig_labels = [10, 20, 30, 40]
        self.labels = list(self.orig_labels)
        self.transform = StableIdentityTransform()
        self.feature_transform = StableIdentityTransform()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        index = int(index)
        type(self).access_log.append(index)
        return self.transform(self.images[index]), self.orig_labels[index]


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

    @staticmethod
    def forward_eval(model, images, indices, cache_size, cache_key="test-dataset"):
        return model.forward_eval(
            images,
            "cpu",
            cache_key=cache_key,
            cache_indices=indices,
            cache_size=cache_size,
        )

    def test_cache_reuses_raw_backbone_features_but_not_projection_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model, backbone = self.make_model(temp_dir)
            images = torch.arange(2 * 3 * 14 * 14, dtype=torch.float32).reshape(2, 3, 14, 14)
            model.eval()

            with mock.patch("builtins.print") as print_mock:
                first = self.forward_eval(model, images, [0, 1], 2)
                second = self.forward_eval(model, images, [0, 1], 2)
            self.assertTrue(torch.allclose(first, second))
            self.assertEqual(backbone.calls, 1)
            print_mock.assert_not_called()

            with torch.no_grad():
                model.fc.bias.add_(torch.tensor([1.0, -1.0]))
            projected_after_head_change = self.forward_eval(model, images, [0, 1], 2)

            self.assertFalse(torch.allclose(second, projected_after_head_change))
            self.assertEqual(backbone.calls, 1)
            self.assertEqual(model.cache_stats()["fully_cached_batches"], 2)
            self.assertEqual(model.cache_stats()["batches_with_misses"], 1)
            self.assertEqual(model.cache_stats()["memory_hit_samples"], 0)
            self.assertEqual(model.cache_stats()["disk_hit_samples"], 4)
            self.assertEqual(model.cache_stats()["written_samples"], 2)
            cache_paths = list(model.cache_dir.glob("*.features.npy"))
            self.assertEqual(len(cache_paths), 1)
            cached_features = np.load(cache_paths[0], mmap_mode="r")
            self.assertEqual(cached_features.shape, (2, retrieval_model.DINOV2_ARCHS["s"]))
            self.assertEqual(cached_features.nbytes, cached_features.size * cached_features.itemsize)
            del cached_features
            self.assertEqual(list(model.cache_dir.glob("*.pt")), [])

    def test_cache_is_shared_across_model_instances(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            images = torch.arange(2 * 3 * 14 * 14, dtype=torch.float32).reshape(2, 3, 14, 14)
            first_model, first_backbone = self.make_model(temp_dir)
            first_model.eval()
            self.forward_eval(first_model, images, [0, 1], 2)
            self.assertEqual(first_backbone.calls, 1)

            second_model, second_backbone = self.make_model(temp_dir)
            second_model.eval()
            self.forward_eval(second_model, images, [0, 1], 2)

            self.assertEqual(second_backbone.calls, 0)
            self.assertEqual(second_model.cache_stats()["hit_samples"], 2)
            self.assertEqual(second_model.cache_stats()["memory_hit_samples"], 0)
            self.assertEqual(second_model.cache_stats()["disk_hit_samples"], 2)

    def test_training_uses_cache_and_backbone_stays_in_eval_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model, backbone = self.make_model(temp_dir)
            images = torch.arange(2 * 3 * 14 * 14, dtype=torch.float32).reshape(2, 3, 14, 14)

            model.train()
            first = model.forward_cached(
                images,
                "cpu",
                cache_key="training",
                cache_indices=[0, 1],
                cache_size=2,
            )
            second = model.forward_cached(
                images,
                "cpu",
                cache_key="training",
                cache_indices=[0, 1],
                cache_size=2,
            )
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

            self.forward_eval(model, images[:2], [0, 1], 3)
            self.forward_eval(model, images[1:], [1, 2], 3)

            self.assertEqual(backbone.calls, 2)
            self.assertEqual(model.cache_stats()["hit_samples"], 1)
            self.assertEqual(model.cache_stats()["miss_samples"], 3)
            self.assertEqual(len(list(model.cache_dir.glob("*.features.npy"))), 1)

    def test_tensor_without_stable_indices_bypasses_persistent_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model, backbone = self.make_model(temp_dir)
            images = torch.arange(2 * 3 * 14 * 14, dtype=torch.float32).reshape(2, 3, 14, 14)

            first = model.forward_eval(images, "cpu")
            second = model.forward_eval(images, "cpu")

            self.assertTrue(torch.allclose(first, second))
            self.assertEqual(backbone.calls, 2)
            self.assertEqual(model.cache_stats()["uncached_samples"], 4)
            self.assertEqual(list(model.cache_dir.glob("*.features.npy")), [])

    def test_precompute_reuses_source_rows_across_different_subsets(self):
        args = SimpleNamespace(
            device="cpu",
            batch_size=2,
            frozen_feature_batch_size=None,
            seed=7,
            num_workers=0,
            dataloader_start_method="spawn",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            first_source = IndexedImageDataset()
            first_subset = Subset(first_source, [0, 1, 2])
            first_subset.orig_labels = [10, 20, 30]
            first_subset.labels = [0, 1, 2]
            first_subset.feature_transform = first_source.feature_transform

            second_source = IndexedImageDataset()
            second_subset = Subset(second_source, [1, 2, 3])
            second_subset.orig_labels = [20, 30, 40]
            second_subset.labels = [0, 1, 2]
            second_subset.feature_transform = second_source.feature_transform

            cache = FrozenFeatureDatasetCache()
            first_model, first_backbone = self.make_model(temp_dir)
            second_model, second_backbone = self.make_model(temp_dir)
            IndexedImageDataset.access_log = []

            first = engine._precompute_backbone_features(
                args,
                first_model,
                first_subset,
                "first indexed subset",
                pin_memory=False,
                require_feature_transform=True,
                frozen_feature_cache=cache,
            )
            second = engine._precompute_backbone_features(
                args,
                second_model,
                second_subset,
                "overlapping indexed subset",
                pin_memory=False,
                require_feature_transform=False,
                frozen_feature_cache=cache,
            )
            full_source = IndexedImageDataset()
            full_subset = Subset(full_source, [0, 1, 2, 3])
            full_subset.orig_labels = [10, 20, 30, 40]
            full_subset.labels = [0, 1, 2, 3]
            full_subset.feature_transform = full_source.feature_transform
            third_model, third_backbone = self.make_model(temp_dir)
            third = engine._precompute_backbone_features(
                args,
                third_model,
                full_subset,
                "fully cached indexed source",
                pin_memory=False,
                require_feature_transform=False,
                frozen_feature_cache=cache,
            )

            self.assertEqual(first_backbone.calls, 2)
            self.assertEqual(second_backbone.calls, 1)
            self.assertEqual(third_backbone.calls, 0)
            self.assertEqual(IndexedImageDataset.access_log, [0, 1, 2, 3])
            self.assertIsNotNone(first.feature_indices)
            self.assertIsNotNone(second.feature_indices)
            self.assertEqual(second_model.cache_stats()["hit_samples"], 2)
            self.assertEqual(second_model.cache_stats()["miss_samples"], 1)
            self.assertEqual(third_model.cache_stats()["hit_samples"], 4)
            self.assertEqual(third_model.cache_stats()["miss_samples"], 0)
            self.assertEqual(len(list(second_model.cache_dir.glob("*.features.npy"))), 1)
            torch.testing.assert_close(
                second.features,
                second_source.images[[1, 2, 3]].flatten(start_dim=1)[
                    :, : retrieval_model.DINOV2_ARCHS["s"]
                ],
            )

            # Windows keeps mapped files locked until every dataset view and
            # coordinator reference has been released.
            del first, second, third, cache, first_model, second_model, third_model
            gc.collect()

    def test_concurrent_overlapping_requests_compute_each_row_once(self):
        barrier = threading.Barrier(2)

        class CoordinatedBackbone(FakeBackbone):
            def forward(self, images):
                barrier.wait(timeout=5)
                return super().forward(images)

        def make_coordinated_model(cache_dir):
            backbone = CoordinatedBackbone()
            with mock.patch.object(
                retrieval_model,
                "load_dinov2_with_retry",
                return_value=backbone,
            ):
                model = retrieval_model.DinoWrapper(
                    dino_size="s",
                    feat_dim=2,
                    backbone_tuning="frozen",
                    use_cache=True,
                    cache_dir=cache_dir,
                )
            return model, backbone

        with tempfile.TemporaryDirectory() as temp_dir:
            first_model, first_backbone = make_coordinated_model(temp_dir)
            second_model, second_backbone = make_coordinated_model(temp_dir)
            second_model.load_state_dict(first_model.state_dict())
            images = torch.arange(
                3 * 3 * 14 * 14,
                dtype=torch.float32,
            ).reshape(3, 3, 14, 14)

            with ThreadPoolExecutor(max_workers=2) as executor:
                first_future = executor.submit(
                    self.forward_eval,
                    first_model,
                    images[:2],
                    [0, 1],
                    3,
                    "concurrent",
                )
                second_future = executor.submit(
                    self.forward_eval,
                    second_model,
                    images[1:],
                    [1, 2],
                    3,
                    "concurrent",
                )
                first = first_future.result(timeout=10)
                second = second_future.result(timeout=10)

            self.assertEqual(first_backbone.calls, 1)
            self.assertEqual(second_backbone.calls, 1)
            self.assertTrue(torch.allclose(first[1], second[0]))
            self.assertEqual(
                first_model.cache_stats()["miss_samples"]
                + second_model.cache_stats()["miss_samples"],
                3,
            )

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
