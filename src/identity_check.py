"""Identity check: how far are the fitted identities from the GNM mean head and
from each other, compared with random identities from the GNM prior?

For every results/<version>/<photo>/gnm_fit.npz:
  - L2 norm and number of values at ±3, per identity part (head 170, eyes 3,
    teeth 80, from GNM's identity_names),
  - mean per-vertex distance (mm) between the identity mesh (expression = 0,
    pose = 0) and the GNM mean head (identity = 0), on all vertices and on the
    face region only (GNM vertex group "hockey_mask": outer face skin, without
    eyes, teeth, ears, neck and back of head),
  - the same distances between every pair of fitted identities.

Baselines, same method:
  - GNM's own sampler (semantic_sampler.IdentitySampler.randomize_identities,
    a conditional VAE; needs TensorFlow), if it can be loaded,
  - N(0, 1) per identity value, clipped to ±3 like the fit.

Writes the "## Identity check" section of results/metrics.md (replacing an
older one). Does not change any fit.

    python src/identity_check.py --version v2-nowarp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from preflight import require

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SECTION = "## Identity check"
FACE_GROUP = "hockey_mask"
PARTS = ("head", "eyes", "teeth")
LIMIT = 3.0


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def distance_to(meshes: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean per-vertex distance (mm) of each mesh to `reference`, over `mask`."""
    return np.linalg.norm(meshes[:, mask] - reference[mask], axis=-1).mean(axis=1) * 1000.0


def pairwise(meshes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(N, N) mean per-vertex distances (mm) over `mask`."""
    m = meshes[:, mask]
    n = len(m)
    out = np.zeros((n, n))
    for i in range(n - 1):
        d = np.linalg.norm(m[i + 1:] - m[i], axis=-1).mean(axis=1) * 1000.0
        out[i, i + 1:] = out[i + 1:, i] = d
    return out


def upper(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(len(matrix), k=1)]


def gnm_sampler_identities(n: int, seed: int) -> tuple[np.ndarray | None, str]:
    try:
        from gnm.shape import semantic_sampler

        sampler = semantic_sampler.IdentitySampler()
        ids = sampler.randomize_identities(num_samples=n, rng=np.random.default_rng(seed))
        return np.asarray(ids, dtype=np.float64), ""
    except Exception as e:  # TensorFlow missing or model not loadable
        return None, f"{type(e).__name__}: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True, help="Folder name under results/, e.g. v2-nowarp")
    parser.add_argument("--results", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--samples", type=int, default=200, help="Random identities per baseline")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    require(gnm=True)
    from gnm.shape import gnm_numpy

    results_dir = Path(args.results)
    fits = sorted((results_dir / args.version).glob("*/gnm_fit.npz"))
    if not fits:
        sys.exit(f"No fits found in {results_dir / args.version}/*/gnm_fit.npz")

    gnm = gnm_numpy.GNM.from_local(
        version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
    mean_head = np.asarray(gnm(None, None, None, None))
    dtype = mean_head.dtype
    all_vertices = np.ones(mean_head.shape[0], dtype=bool)
    face = np.asarray(gnm.vertex_groups)[list(gnm.vertex_group_names).index(FACE_GROUP)] > 0.5
    part_index = {p: np.array([n.startswith(p + "_") for n in gnm.identity_names]) for p in PARTS}

    def meshes_for(ids: np.ndarray) -> np.ndarray:
        return np.asarray(gnm(ids.astype(dtype), None, None, None))

    names = [f.parent.name for f in fits]
    fit_ids = np.stack([np.load(f)["identity"].reshape(-1) for f in fits]).astype(np.float64)
    fit_meshes = meshes_for(fit_ids)

    # ---- per fit -----------------------------------------------------------
    fit_all = distance_to(fit_meshes, mean_head, all_vertices)
    fit_face = distance_to(fit_meshes, mean_head, face)
    fit_pair_all = pairwise(fit_meshes, all_vertices)
    fit_pair_face = pairwise(fit_meshes, face)

    rows = []
    for i, name in enumerate(names):
        row = [name, f"{np.linalg.norm(fit_ids[i]):.2f}"]
        for p in PARTS:
            x = fit_ids[i, part_index[p]]
            row += [f"{np.linalg.norm(x):.2f}", int(np.isclose(np.abs(x), LIMIT, atol=1e-4).sum())]
        row += [f"{fit_all[i]:.2f}", f"{fit_face[i]:.2f}"]
        rows.append(row)
    sizes = {p: int(part_index[p].sum()) for p in PARTS}
    per_fit = md_table(
        ["Photo", "L2 all", *(h for p in PARTS for h in (f"L2 {p}", f"{p} at ±3 (/{sizes[p]})")),
         "To mean head, all (mm)", "To mean head, face (mm)"],
        rows,
    )

    # ---- baselines ---------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    groups = [(f"Fits ({args.version})", len(names), fit_ids, fit_meshes)]
    sampler_ids, sampler_error = gnm_sampler_identities(args.samples, args.seed)
    if sampler_ids is not None:
        groups.append(("GNM IdentitySampler", args.samples, sampler_ids, meshes_for(sampler_ids)))
    normal_ids = np.clip(rng.standard_normal((args.samples, fit_ids.shape[1])), -LIMIT, LIMIT)
    groups.append(("N(0,1), clipped ±3", args.samples, normal_ids, meshes_for(normal_ids)))

    summary_rows = []
    for label, n, ids, meshes in groups:
        at_limit = np.isclose(np.abs(ids), LIMIT, atol=1e-4).sum(axis=1)
        summary_rows.append([
            label, n,
            f"{np.linalg.norm(ids, axis=1).mean():.2f}",
            *(f"{np.linalg.norm(ids[:, part_index[p]], axis=1).mean():.2f}" for p in PARTS),
            f"{at_limit.mean():.1f}",
            f"{distance_to(meshes, mean_head, all_vertices).mean():.2f}",
            f"{distance_to(meshes, mean_head, face).mean():.2f}",
            f"{upper(pairwise(meshes, all_vertices)).mean():.2f}",
            f"{upper(pairwise(meshes, face)).mean():.2f}",
        ])
    summary = md_table(
        ["Set", "N", "L2 all", *(f"L2 {p}" for p in PARTS), "Values at ±3",
         "To mean head, all (mm)", "To mean head, face (mm)",
         "Pairwise, all (mm)", "Pairwise, face (mm)"],
        summary_rows,
    )

    pair_table = md_table(
        ["all / face (mm)", *names],
        [[name, *("–" if i == j else f"{fit_pair_all[i, j]:.2f} / {fit_pair_face[i, j]:.2f}"
                  for j in range(len(names)))]
         for i, name in enumerate(names)],
    )

    sampler_note = (
        "GNM IdentitySampler = `semantic_sampler.IdentitySampler().randomize_identities()` "
        "(conditional VAE with random gender / ethnicity weights; its output is not clipped)."
        if sampler_ids is not None
        else f"GNM IdentitySampler could not be loaded ({sampler_error}); only the N(0,1) baseline is shown."
    )
    section = (
        f"{SECTION}\n\n"
        f"Landmark version: {args.version}. Each identity is evaluated with "
        "expression = 0 and pose = 0 and compared with the GNM mean head "
        "(identity = 0). Distance = mean Euclidean distance per vertex, in mm, "
        f"over all {int(all_vertices.sum())} vertices or over the {int(face.sum())} "
        f"face vertices (GNM vertex group `{FACE_GROUP}`: outer face skin, no eyes, "
        "teeth, ears, neck or back of head). Identity parts follow GNM's "
        f"identity names: head {sizes['head']}, eyes {sizes['eyes']}, "
        f"teeth {sizes['teeth']}.\n\n"
        f"### Fits vs random identities\n\n{summary}\n\n"
        f"{sampler_note} Random samples: seed {args.seed}. Means over the set; "
        "pairwise = mean over all pairs.\n\n"
        f"### Per fit\n\n{per_fit}\n\n"
        f"### Pairwise distance between fitted identities\n\n{pair_table}\n"
    )

    md_path = results_dir / "metrics.md"
    text = md_path.read_text() if md_path.is_file() else "# GNM fit metrics\n"
    md_path.write_text(text.split(SECTION)[0].rstrip() + "\n\n" + section)

    print(summary, per_fit, pair_table, sep="\n\n")
    if sampler_error:
        print(f"\nIdentitySampler not used: {sampler_error}")
    print(f"\nWrote {md_path}")


if __name__ == "__main__":
    main()
