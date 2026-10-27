"""The two t-SNE / LP frames required by fig_lp_cars, from the data bundle."""
import datensammlung as sammlung


def run_dir(variant, k=16, fold="00"):
    if (variant, k, fold) != ("iscen_cir", 16, "00"):
        raise ValueError("The compact bundle includes iscen_cir, k=16, fold=00.")
    return "qualitative"


def epochs(tdir):
    return sammlung.read_json("qualitative/lp_epochs.json")


def frame(tdir, ep):
    r = sammlung.read_csv(f"qualitative/lp_{ep}.csv", float_precision="round_trip")
    types = sammlung.manifest()["views"][f"qualitative/lp_{ep}.csv"]["array_dtypes"]
    # Original confidences use float16. Both their values and threshold
    # comparisons depend on retaining this dtype after a CSV round trip.
    return {c: r[c].to_numpy().astype(types[c]) for c in r}
