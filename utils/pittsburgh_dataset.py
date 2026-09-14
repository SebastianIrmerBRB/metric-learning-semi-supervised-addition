"""Pittsburgh 30k/250k visual place recognition datasets.

Both benchmarks come from the same Google Street View collection released with
NetVLAD (Arandjelovic et al., CVPR 2016). A *place* is one panorama, and the
released images are its 24 perspective crops -- 12 yaw directions at two
pitches. Queries are crops of panoramas photographed at the same locations in
different years, which is where the temporal variation comes from; they live in
their own directory and never appear in the database.

That structure is what makes the collection usable as a metric-learning dataset
here: the panorama is the class. Every database crop is labelled with its own
panorama, and every query crop is labelled with the *nearest database panorama
within ``posDistThr``* (25 m, read from the split metadata). Queries with no
database panorama inside that radius are unanswerable under a class-based
retrieval protocol and are dropped, with the count recorded.

Note what that costs relative to the usual VPR protocol. Standard Recall@N on
Pittsburgh counts a retrieval correct if *any* database image within 25 m of the
query is returned; here only crops of the single nearest panorama count, so
``precision_at_1`` is strictly harsher than published Recall@1 and the two are
not comparable. ``geographic_positive_labels`` returns the full radius-based
positive list for anyone who wants to evaluate the published way instead.

Expected local layout under ``root`` is the raw NetVLAD distribution::

    000/ ... 010/            database crops, ``<pano_id>_pitch<p>_yaw<y>.jpg``
    queries_real/            query crops, same naming
    datasets/                pitts30k_{train,val,test}.mat
                             pitts250k_{train,val,test}.mat

Pitts30k is a geographic subsample of the same image tree, so a single copy
serves both dataset names: if ``data/Pittsburgh30k`` holds no metadata, the
sibling directories named in ``SHARED_ROOT_NAMES`` are searched as well.

The collection is released only on request, so it is never downloaded
automatically.
"""

from pathlib import Path, PurePosixPath

import numpy as np
from loguru import logger
from scipy.io import loadmat
from scipy.spatial import cKDTree
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class _Pittsburgh(Dataset):
    """Shared loader for the Pitts30k and Pitts250k splits.

    ``split`` accepts the three official geographic splits, the pooled
    ``train+val`` development half that ``dataset_protocol=official`` uses, and
    ``train+val+test`` for inspection. Every split returns its query crops first
    and its database crops second, and exposes ``query_indices`` /
    ``gallery_indices`` so the final test split is scored query-to-gallery.
    """

    SIZE_SUFFIX = ""
    OFFICIAL_SPLITS = ("train", "val", "test")
    AVAILABLE_SPLITS = ("train", "val", "test", "train+val", "train+val+test")
    # Which split ``dataset_protocol=official`` loads as the development pool.
    # Pittsburgh ships a validation split of its own, but this project carves
    # validation out of the development pool itself, so both halves are pooled
    # and re-split rather than leaving a third of the data unused.
    OFFICIAL_DEVELOPMENT_SPLIT = "train+val"
    METADATA_DIRECTORY = "datasets"
    QUERY_DIRECTORY = "queries_real"
    # Sibling directory names searched when ``root`` itself holds no metadata,
    # so one extracted copy of the 250k tree can serve both dataset names.
    SHARED_ROOT_NAMES = ("Pittsburgh", "pittsburgh", "Pittsburgh250k", "pitts250k")
    # Published split sizes as ``(database_images, query_images)``. The Pitts30k
    # rows and the Pitts250k test row are the numbers NetVLAD reports; the
    # Pitts250k train and val rows are derived from its published totals of
    # 254,064 database and 23,712 query images. A mismatch is reported rather
    # than raised unless ``strict_expected_counts`` asks for it.
    EXPECTED_SPLIT_COUNTS = {}
    # Two field spellings for the same MATLAB struct are in circulation.
    DB_IMAGE_FIELDS = ("dbImageFns", "dbImage", "dbImageFn")
    QUERY_IMAGE_FIELDS = ("qImageFns", "qImage", "qImageFn")
    DB_UTM_FIELDS = ("utmDb",)
    QUERY_UTM_FIELDS = ("utmQ",)
    POSITION_THRESHOLD_FIELDS = ("posDistThr",)
    # All 24 crops of one panorama share its UTM position. A pano id that
    # resolves to positions further apart than this is a namespace collision,
    # not a panorama, and would silently merge two places into one class.
    MAX_PLACE_UTM_SPREAD_M = 1.0

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        target_transform=None,
        download=False,
        place_radius_m=None,
        strict_expected_counts=False,
    ):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        if split not in self.AVAILABLE_SPLITS:
            raise ValueError(f"split must be one of {self.AVAILABLE_SPLITS}, got {split!r}")
        self.split = split

        dataset_root = self.find_dataset_root(self.root)
        if dataset_root is None:
            if download:
                raise RuntimeError(
                    f"Pitts{self.SIZE_SUFFIX} does not support automatic download; the "
                    "Pittsburgh collection is released only on request. Prepare it "
                    f"manually under {self.root} with the database directories 000/ ... "
                    f"010/, {self.QUERY_DIRECTORY}/, and "
                    f"{self.METADATA_DIRECTORY}/pitts{self.SIZE_SUFFIX}_{{train,val,test}}.mat."
                )
            raise ValueError(
                f"Pitts{self.SIZE_SUFFIX} metadata was not found. Expected "
                f"{self.METADATA_DIRECTORY}/pitts{self.SIZE_SUFFIX}_train.mat and a "
                f"{self.QUERY_DIRECTORY}/ directory below {self.root} or one of "
                f"{', '.join(self.SHARED_ROOT_NAMES)} beside it."
            )
        self.dataset_root = dataset_root

        query_records = []
        database_records = []
        split_counts = {}
        place_centroids = []
        place_labels = []
        radii = []
        for official_split in self.official_splits(split):
            block = self._load_official_split(
                official_split,
                place_radius_m=place_radius_m,
                strict_expected_counts=strict_expected_counts,
            )
            query_records.extend(block["query_records"])
            database_records.extend(block["database_records"])
            split_counts[official_split] = block["counts"]
            place_centroids.append(block["place_centroids"])
            place_labels.append(block["place_labels"])
            radii.append(block["place_radius_m"])

        records = query_records + database_records
        if not records:
            raise ValueError(f"Pitts{self.SIZE_SUFFIX} split {split!r} is empty")
        self._require_readable_images(records)

        self.records = records
        self.paths = [record["path"] for record in records]
        self.image_names = [record["image_name"] for record in records]
        self.pano_ids = [record["pano_id"] for record in records]
        self.place_keys = [record["place_key"] for record in records]
        self.labels = [record["label"] for record in records]
        self.orig_labels = list(self.labels)
        self.is_query = np.asarray([record["is_query"] for record in records], dtype=bool)
        self.utm = np.asarray([record["utm"] for record in records], dtype=np.float64)
        self.query_indices = list(range(len(query_records)))
        self.gallery_indices = list(range(len(query_records), len(records)))

        # Place geometry kept for radius-based (published-protocol) evaluation
        # and for pseudo-labelling that wants to reason about distance rather
        # than about the single nearest panorama.
        self.place_centroids = np.concatenate(place_centroids, axis=0)
        self.place_labels = np.concatenate(place_labels, axis=0)
        self.place_radius_m = float(min(radii))
        self.class_to_label = {
            record["place_key"]: record["label"] for record in database_records
        }
        self.classes = [
            place_key
            for place_key, _ in sorted(self.class_to_label.items(), key=lambda item: item[1])
        ]

        # The three official splits are cut geographically, so no panorama --
        # and therefore no class -- appears in more than one of them.
        self.class_disjoint_split = True
        self.class_split_info = self._build_class_split_info(split_counts)

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------

    @classmethod
    def official_splits(cls, split):
        """Expand a requested split into the official splits it pools."""

        return tuple(part for part in split.split("+"))

    @classmethod
    def metadata_filename(cls, official_split):
        return f"pitts{cls.SIZE_SUFFIX}_{official_split}.mat"

    @classmethod
    def find_dataset_root(cls, root):
        """Locate the raw NetVLAD tree, or return None when it is absent."""

        root = Path(root)
        named = [root, root / "pitts250k", root / "pitts30k", root / "Pittsburgh"]
        named.extend(root.parent / name for name in cls.SHARED_ROOT_NAMES)

        # A named directory wins over anything below it, so each one is queued
        # ahead of its own children; the children are searched because the
        # archive commonly extracts one directory deep.
        candidates = []
        for candidate in named:
            if not candidate.is_dir():
                continue
            candidates.append(candidate)
            candidates.extend(child for child in sorted(candidate.iterdir()) if child.is_dir())

        seen = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if not (candidate / cls.QUERY_DIRECTORY).is_dir():
                continue
            metadata_directory = candidate / cls.METADATA_DIRECTORY
            if (metadata_directory / cls.metadata_filename("train")).is_file():
                return candidate
        return None

    @classmethod
    def is_ready(cls, root):
        dataset_root = cls.find_dataset_root(root)
        if dataset_root is None:
            return False
        metadata_directory = dataset_root / cls.METADATA_DIRECTORY
        return all(
            (metadata_directory / cls.metadata_filename(official_split)).is_file()
            for official_split in cls.OFFICIAL_SPLITS
        )

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    @classmethod
    def parse_db_struct(cls, metadata_path):
        """Read one ``dbStruct`` .mat into plain arrays.

        MATLAB stores the coordinates as ``2 x N``; they are transposed here so
        every array is indexed by image.
        """

        contents = loadmat(str(metadata_path), struct_as_record=False, squeeze_me=True)
        if "dbStruct" not in contents:
            raise ValueError(
                f"{metadata_path} contains no dbStruct variable; found "
                f"{sorted(key for key in contents if not key.startswith('__'))}"
            )
        db_struct = contents["dbStruct"]

        database_names = cls._string_field(db_struct, cls.DB_IMAGE_FIELDS, metadata_path)
        query_names = cls._string_field(db_struct, cls.QUERY_IMAGE_FIELDS, metadata_path)
        database_utm = cls._utm_field(
            db_struct, cls.DB_UTM_FIELDS, len(database_names), metadata_path
        )
        query_utm = cls._utm_field(db_struct, cls.QUERY_UTM_FIELDS, len(query_names), metadata_path)
        position_threshold = float(
            cls._named_field(db_struct, cls.POSITION_THRESHOLD_FIELDS, metadata_path)
        )
        if not position_threshold > 0:
            raise ValueError(
                f"{metadata_path} declares a non-positive posDistThr: {position_threshold}"
            )
        return {
            "database_names": database_names,
            "database_utm": database_utm,
            "query_names": query_names,
            "query_utm": query_utm,
            "position_threshold_m": position_threshold,
        }

    @staticmethod
    def _named_field(db_struct, names, metadata_path):
        for name in names:
            if hasattr(db_struct, name):
                return getattr(db_struct, name)
        available = ", ".join(sorted(getattr(db_struct, "_fieldnames", ())))
        raise ValueError(
            f"{metadata_path} dbStruct has none of the fields {names}; it holds {available}"
        )

    @classmethod
    def _string_field(cls, db_struct, names, metadata_path):
        values = np.atleast_1d(cls._named_field(db_struct, names, metadata_path))
        return [str(value).strip() for value in values.ravel()]

    @classmethod
    def _utm_field(cls, db_struct, names, expected_count, metadata_path):
        values = np.asarray(cls._named_field(db_struct, names, metadata_path), dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(
                f"{metadata_path} field {names[0]} must be two-dimensional, got shape {values.shape}"
            )
        # MATLAB stores these 2 x N, so that reading wins whenever it fits --
        # including the 2 x 2 case where both orientations would match.
        if values.shape[0] == 2 and values.shape[1] == expected_count:
            return values.T
        if values.shape[1] == 2 and values.shape[0] == expected_count:
            return values
        raise ValueError(
            f"{metadata_path} field {names[0]} has shape {values.shape}, which matches "
            f"neither ({expected_count}, 2) nor (2, {expected_count})"
        )

    # ------------------------------------------------------------------
    # record construction
    # ------------------------------------------------------------------

    def _load_official_split(self, official_split, place_radius_m, strict_expected_counts):
        if official_split not in self.OFFICIAL_SPLITS:
            raise ValueError(
                f"{official_split!r} is not one of the official splits {self.OFFICIAL_SPLITS}"
            )
        metadata_path = (
            self.dataset_root / self.METADATA_DIRECTORY / self.metadata_filename(official_split)
        )
        if not metadata_path.is_file():
            raise ValueError(
                f"Pitts{self.SIZE_SUFFIX} split {official_split!r} needs {metadata_path}, "
                "which is missing"
            )
        struct = self.parse_db_struct(metadata_path)
        self._check_expected_counts(
            official_split,
            database_count=len(struct["database_names"]),
            query_count=len(struct["query_names"]),
            metadata_path=metadata_path,
            strict_expected_counts=strict_expected_counts,
        )

        place_keys, place_labels, place_centroids, place_index, utm_spread = self._group_places(
            struct["database_names"],
            struct["database_utm"],
            metadata_path,
        )
        radius = (
            struct["position_threshold_m"] if place_radius_m is None else float(place_radius_m)
        )
        if not radius > 0:
            raise ValueError(f"place_radius_m must be positive, got {place_radius_m!r}")

        database_records = [
            {
                "image_name": name,
                "path": str(self._resolve_database_path(name, metadata_path)),
                "pano_id": place_keys[place_index[position]],
                "place_key": place_keys[place_index[position]],
                "label": int(place_labels[place_index[position]]),
                "utm": struct["database_utm"][position],
                "is_query": False,
                "official_split": official_split,
            }
            for position, name in enumerate(struct["database_names"])
        ]

        matched_places, unmatched_count = self._match_queries_to_places(
            struct["query_utm"],
            place_centroids,
            radius,
        )
        query_records = [
            {
                "image_name": name,
                "path": str(self._resolve_query_path(name, metadata_path)),
                "pano_id": self.pano_id_of(name),
                "place_key": place_keys[matched_places[position]],
                "label": int(place_labels[matched_places[position]]),
                "utm": struct["query_utm"][position],
                "is_query": True,
                "official_split": official_split,
            }
            for position, name in enumerate(struct["query_names"])
            if matched_places[position] >= 0
        ]
        if not query_records:
            raise ValueError(
                f"Pitts{self.SIZE_SUFFIX} split {official_split!r} matched none of its "
                f"{len(struct['query_names'])} queries to a database panorama within "
                f"{radius:g} m; check that {metadata_path} belongs to this collection"
            )

        return {
            "query_records": query_records,
            "database_records": database_records,
            "place_centroids": place_centroids,
            "place_labels": place_labels,
            "place_radius_m": radius,
            "counts": {
                "database_images": len(database_records),
                "query_images": len(query_records),
                "released_query_images": len(struct["query_names"]),
                "unmatched_query_images": int(unmatched_count),
                "places": int(len(place_keys)),
                "position_distance_threshold_m": float(radius),
                "max_place_utm_spread_m": float(utm_spread),
            },
        }

    @classmethod
    def _group_places(cls, database_names, database_utm, metadata_path):
        """Collapse the crops of each panorama into one place.

        The panorama id doubles as the class label, which keeps labels stable
        across splits without reading every split's metadata first.
        """

        pano_ids = np.asarray([cls.pano_id_of(name) for name in database_names])
        non_numeric = sorted({pano_id for pano_id in pano_ids.tolist() if not pano_id.isdigit()})
        if non_numeric:
            raise ValueError(
                f"{metadata_path} lists database images whose panorama id is not numeric, so it "
                f"cannot be used as a stable class label: {non_numeric[:5]}"
            )

        place_keys, place_index = np.unique(pano_ids, return_inverse=True)
        place_index = place_index.reshape(-1)
        centroid_sums = np.zeros((len(place_keys), 2), dtype=np.float64)
        np.add.at(centroid_sums, place_index, database_utm)
        counts = np.bincount(place_index, minlength=len(place_keys)).astype(np.float64)
        centroids = centroid_sums / counts[:, None]

        spread = float(np.linalg.norm(database_utm - centroids[place_index], axis=1).max())
        if spread > cls.MAX_PLACE_UTM_SPREAD_M:
            raise ValueError(
                f"{metadata_path} maps one panorama id onto positions {spread:.1f} m apart, "
                f"more than the {cls.MAX_PLACE_UTM_SPREAD_M:g} m a single panorama allows; "
                "the ids are not unique and would merge distinct places into one class"
            )

        place_labels = np.asarray([int(place_key) for place_key in place_keys], dtype=np.int64)
        return place_keys.tolist(), place_labels, centroids, place_index, spread

    @staticmethod
    def _match_queries_to_places(query_utm, place_centroids, radius):
        """Label each query with its nearest database panorama inside ``radius``.

        Returns a place index per query, ``-1`` where nothing is in range.
        """

        if len(query_utm) == 0:
            return np.zeros(0, dtype=np.int64), 0
        tree = cKDTree(place_centroids)
        distances, indices = tree.query(query_utm, k=1, distance_upper_bound=radius)
        matched = np.isfinite(distances)
        place_index = np.where(matched, indices, -1).astype(np.int64)
        return place_index, int((~matched).sum())

    @staticmethod
    def pano_id_of(image_name):
        """Take the panorama id from ``<pano_id>_pitch<p>_yaw<y>[_<note>].jpg``."""

        return PurePosixPath(str(image_name).replace("\\", "/")).name.split("_")[0]

    @classmethod
    def _validated_relative_path(cls, image_name, metadata_path):
        relative_path = PurePosixPath(str(image_name).replace("\\", "/"))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"{metadata_path} lists an unsafe image path: {image_name!r}")
        return Path(relative_path)

    def _resolve_database_path(self, image_name, metadata_path):
        relative_path = self._validated_relative_path(image_name, metadata_path)
        return self.dataset_root / relative_path

    def _resolve_query_path(self, image_name, metadata_path):
        relative_path = self._validated_relative_path(image_name, metadata_path)
        candidates = [
            self.dataset_root / self.QUERY_DIRECTORY / relative_path,
            self.dataset_root / relative_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _require_readable_images(self, records):
        """Probe a few records rather than stat 250k files on every load.

        Queries come first and the database second, so sampling both ends and
        the middle covers a query and a database crop whatever the split holds.
        """

        probes = sorted({0, len(records) // 2, len(records) - 1})
        if any(Path(records[position]["path"]).exists() for position in probes):
            return
        raise ValueError(
            f"Pitts{self.SIZE_SUFFIX} images were not found below {self.dataset_root}. Expected "
            f"the database directories named in {self.METADATA_DIRECTORY}/ and a "
            f"{self.QUERY_DIRECTORY}/ directory holding the query crops."
        )

    def _check_expected_counts(
        self,
        official_split,
        database_count,
        query_count,
        metadata_path,
        strict_expected_counts,
    ):
        expected = self.EXPECTED_SPLIT_COUNTS.get(official_split)
        if expected is None:
            return
        expected_database, expected_query = expected
        if (database_count, query_count) == (expected_database, expected_query):
            return
        message = (
            f"Pitts{self.SIZE_SUFFIX} split {official_split!r} holds {database_count} database "
            f"and {query_count} query images; the published split has {expected_database} and "
            f"{expected_query}. Check that {metadata_path} is the released dbStruct."
        )
        if strict_expected_counts:
            raise ValueError(message)
        logger.warning(message)

    def _build_class_split_info(self, split_counts):
        development_splits = self.official_splits(self.OFFICIAL_DEVELOPMENT_SPLIT)
        return {
            "dataset_root": str(self.dataset_root),
            "pittsburgh_size": self.SIZE_SUFFIX,
            "split": self.split,
            "split_basis": "official_geographic_splits",
            "class_disjoint_test": True,
            "official_image_level_split_used": True,
            "development_splits": list(development_splits),
            "held_out_test_split": "test",
            "place_definition": "street_view_panorama",
            "query_label_mode": "nearest_place_within_pos_dist_thr",
            "position_distance_threshold_m": self.place_radius_m,
            "place_count": int(len(self.class_to_label)),
            "database_image_count": int(len(self.gallery_indices)),
            "query_image_count": int(len(self.query_indices)),
            "unmatched_query_image_count": int(
                sum(counts["unmatched_query_images"] for counts in split_counts.values())
            ),
            "max_place_utm_spread_m": float(
                max(counts["max_place_utm_spread_m"] for counts in split_counts.values())
            ),
            "official_split_counts": split_counts,
        }

    # ------------------------------------------------------------------
    # public helpers
    # ------------------------------------------------------------------

    def geographic_positive_labels(self, radius_m=None):
        """Every place label within ``radius_m`` of each query, in query order.

        This is the ground truth the published Pittsburgh Recall@N uses. The
        ``labels`` this dataset trains and scores on keep only the nearest of
        these, so a run that wants the published protocol has to score against
        this list instead.
        """

        radius = self.place_radius_m if radius_m is None else float(radius_m)
        if not radius > 0:
            raise ValueError(f"radius_m must be positive, got {radius_m!r}")
        tree = cKDTree(self.place_centroids)
        neighbourhoods = tree.query_ball_point(self.utm[self.query_indices], r=radius)
        return [
            sorted(int(self.place_labels[position]) for position in neighbourhood)
            for neighbourhood in neighbourhoods
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image = default_loader(self.paths[index])
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label


class Pittsburgh30k(_Pittsburgh):
    """Pitts30k: 30,000 database and 21,840 query crops over train/val/test."""

    SIZE_SUFFIX = "30k"
    EXPECTED_SPLIT_COUNTS = {
        "train": (10_000, 7_416),
        "val": (10_000, 7_608),
        "test": (10_000, 6_816),
    }


class Pittsburgh250k(_Pittsburgh):
    """Pitts250k: 254,064 database and 23,712 query crops over train/val/test."""

    SIZE_SUFFIX = "250k"
    EXPECTED_SPLIT_COUNTS = {
        "train": (91_464, 7_824),
        "val": (78_648, 7_608),
        "test": (83_952, 8_280),
    }
