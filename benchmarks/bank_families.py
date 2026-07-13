#!/usr/bin/env python3
"""The counting-benchmark bank SPEC — single source of truth for every family.

Committed and hand-authored from the survey
(docs/research/2026-07-13-counting-benchmark-survey.md, which is the contract for
WHAT goes in the bank). Both the `fetch_*.py` scripts and `bank_manifest_gen.py`
import this, so a URL / license / conversion is stated in exactly one place.

Two kinds of family:
  * "external"  — fetched from a remote archive (Zenodo/GitHub/host), then
                  converted to wcn-1 on demand. `source` records how to fetch.
  * "generator" — produced by an in-repo generator; the generator + seed grid IS
                  the provenance (no pre-generated data committed).

Field notes for external families:
  license       SPDX id or short tag; "as-is" = no OSI license (Cachet lineage).
  fmt           on-disk format before conversion.
  conversion    the converter that lands it in wcn-1 (or "deferred:M3").
  reference     how per-instance ground truth is obtained for bank_check.py.
  deferred      True => archive-now only; needs M3 (projected counting) to run.
  zenodo        Zenodo record id (files discovered via API at fetch time).
  urls          explicit download URLs (non-Zenodo sources).
  tracks        for MCC: which `c t` header classes to keep.
"""

# --- external families (priority order mirrors survey §9) -------------------

EXTERNAL = [
    {
        "id": "mcc-2023",
        "priority": 1,
        "title": "Model Counting Competition 2023 — tracks 1 (mc) + 2 (wmc)",
        "zenodo": "10012864",
        "urls": [],
        "license": "CC-BY-4.0",
        "fmt": "DIMACS-CNF (MCC 2021+ headers: `c t mc|wmc`, `c p weight`)",
        "conversion": "to_wcn (Instance::from_dimacs_text)",
        "tracks": ["mc", "wmc"],
        "reference": {
            "kind": "mcc-log10",
            "note": "MCC reference counts are log10 estimates; compare within a "
                    "log10 relative tolerance, never exact. Reference values ship "
                    "with the competition results, not always in the instance "
                    "archive — bank_check.py looks for a sibling *.log10 / results "
                    "file and cross-checks small instances with sharpSAT-TD.",
        },
        "deferred": False,
        "notes": "Reviewer-mandatory. sharpSAT-TD/Ganak/d4 have published numbers "
                 "on this. Archive holds all 4 tracks; fetch keeps only mc+wmc "
                 "(classified by the `c t` header, layout-independent).",
    },
    {
        "id": "mcc-2024",
        "priority": 1,
        "title": "Model Counting Competition 2024 — tracks 1 (mc) + 2 (wmc)",
        "zenodo": "14249068",
        "urls": [],
        "license": "CC-BY-4.0",
        "fmt": "DIMACS-CNF (MCC format v1.1)",
        "conversion": "to_wcn (Instance::from_dimacs_text)",
        "tracks": ["mc", "wmc"],
        "reference": {
            "kind": "mcc-log10",
            "note": "As mcc-2023. Track 2b (negative/zero weights) is OUT OF "
                    "SCOPE for our nonnegative-rational semiring and is skipped.",
        },
        "deferred": False,
        "notes": "415.8 MB compressed but 60.3 GB uncompressed ACROSS ALL 5 "
                 "TRACKS. fetch_mcc.py selectively extracts only mc+wmc members "
                 "(never the full archive) to respect the 12 GB bank cap.",
    },
    {
        "id": "uai-pr",
        "priority": 2,
        "title": "UAI inference competition — PR (partition function) corpus",
        "zenodo": None,
        "urls": [
            # dechterlab mirror (MIT): the 2006/2008/2012/2014 .uai + .uai.evid.
            "https://github.com/dechterlab/uai-competitions/archive/refs/heads/master.tar.gz",
            # UCI 2022 final + tuning (= UAI-2014) PR sets.
            "https://ics.uci.edu/~dechter/uaicompetition/2022/FinalBenchmarks/PR.zip",
            "https://ics.uci.edu/~dechter/uaicompetition/2022/TuningBenchmarks/PR.zip",
        ],
        "license": "MIT (dechterlab mirror)",
        "fmt": ".uai MARKOV (dense non-negative-real factor tables) + .uai.evid",
        "conversion": "uai_to_wcn.py (BINARY-domain instances only; multi-valued "
                      "SKIPPED, not log-encoded — survey gap #2)",
        "tracks": None,
        "reference": {
            "kind": "uai-log10-Z",
            "note": "PR ground truth is log10 Z in a companion *.PR / *.uai.PR "
                    "solution file where present; our exact rational Z is compared "
                    "as log10 within tolerance.",
        },
        "deferred": False,
        "notes": "Native-table home turf; zero encoding loss. Convertible subset = "
                 "all-binary-cardinality instances (Grids/Ising subsets); "
                 "multi-valued (linkage/Promedus/protein) skipped with a manifest "
                 "note. Convertible/skipped tallies filled in at fetch time.",
    },
    {
        "id": "addmc-1914",
        "priority": 3,
        "title": "ADDMC 1,914-instance weighted suite (Cachet-Bayes + CRIL-Non-Bayes)",
        "zenodo": None,
        "urls": [
            "https://github.com/vardigroup/ADDMC/releases/download/v1.0.0/benchmarks.zip",
        ],
        "license": "as-is (Cachet lineage; contains no OSI license)",
        "fmt": "DIMACS-CNF + literal weights (synthetic 0.5/1.5 on Non-Bayes)",
        "conversion": "to_wcn (Instance::from_dimacs_text)",
        "tracks": None,
        "reference": {
            "kind": "none-per-instance",
            "note": "No per-instance reference counts published; ADDMC/DPMC papers "
                    "report aggregates only. Verify by cross-checking small "
                    "instances against sharpSAT-TD.",
        },
        "deferred": False,
        "notes": "Apples-to-apples with every DP-side baseline's published numbers "
                 "(ADDMC/DPMC/TensorOrder2 all evaluate on this exact set). The "
                 "Rochester source URL is dead; this v1.0.0 release asset is live.",
    },
    {
        "id": "kr2024-mc",
        "priority": 4,
        "title": "Model Counting in the Wild (KR 2024) — model-counting archive "
                 "(incl. 411 crypto instances)",
        "zenodo": "13284882",
        "urls": [],
        "license": "CC-BY-4.0",
        "fmt": "DIMACS-CNF (plain #SAT / WMC)",
        "conversion": "to_wcn (Instance::from_dimacs_text)",
        "tracks": None,
        "reference": {"kind": "none-per-instance",
                      "note": "Cross-check small instances with sharpSAT-TD."},
        "deferred": False,
        "only_files": ["model_counting_benchmarks.zip"],
        "notes": "The 411 chosen-ciphertext CRYPTO instances (Beck-Zinkus-Green, "
                 "USENIX 2020) live inside model_counting_benchmarks.zip — the one "
                 "citable crypto-counting family, closing the crypto omission. "
                 "XOR-hard: a γ-fingerprint predicted-hard target.",
    },
    {
        "id": "kr2024-projected",
        "priority": 4,
        "title": "Model Counting in the Wild (KR 2024) — projected archive (117 QIF + more)",
        "zenodo": "13284882",
        "urls": [],
        "license": "CC-BY-4.0",
        "fmt": "DIMACS-CNF with `c p show` projection headers",
        "conversion": "deferred:M3",
        "tracks": None,
        "reference": {"kind": "deferred"},
        "deferred": True,
        "only_files": ["projected_counting_benchmarks.zip"],
        "notes": "ARCHIVE-NOW while the link is alive; 1.4 GB. Needs projected "
                 "counting (M3): our parser correctly REJECTS `c p show` today.",
    },
    {
        "id": "klebanov-qif",
        "priority": 4,
        "title": "Klebanov QIF benchmarks (qif-cnf.tgz)",
        "zenodo": None,
        "urls": [
            "https://formal.kastel.kit.edu/~klebanov/software/qif-cnf.tgz",
        ],
        "license": "unspecified (research use)",
        "fmt": "projected #SAT CNF (leakage = projected count)",
        "conversion": "deferred:M3",
        "tracks": None,
        "reference": {"kind": "deferred"},
        "deferred": True,
        "notes": "ARCHIVE-NOW: the companion ApproxMC-p solver dir already 404'd; "
                 "only this instance tarball survives. Deferred until M3.",
    },
]

# --- generator families (in-repo; generator + seed grid = provenance) -------

GENERATOR = [
    {
        "id": "factoring",
        "mechanism": "arithmetic-circuit",
        "generator": "benchmarks/gen_factoring.py",
        "fmt": "CircuitSAT JSON (problem-reductions)",
        "conversion": "load_instance (from_circuit_sat_json)",
        "grid": {"sizes": [16, 20, 24, 28], "records_per_size": 10},
        "counting_role": "#factorizations / circuit counting; arithmetic axis; "
                         "kin to ISCAS c6288.",
        "seed_note": "RNG seeded per size (SEED_BASE + n); line 0 pinned.",
    },
    {
        "id": "tseitin",
        "mechanism": "parity-treewidth",
        "generator": "benchmarks/csp/gen_tseitin.py",
        "fmt": "native .csp + DIMACS .cnf",
        "conversion": "load_instance (.csp or .cnf)",
        "grid": {"grid_rows": [2, 3, 4, 5], "grid_cols": 40, "grid_seeds": 10,
                 "regular_v": [60, 70, 80, 90, 100], "regular_d": 3,
                 "regular_seeds": 20},
        "counting_role": "parity-counting axis — the γ-fingerprint kill-test "
                         "family; extend toward LFSR/hash-round counting.",
        "seed_note": "grid=predicted win; 3-regular expander=predicted loss.",
    },
    {
        "id": "xor",
        "mechanism": "parity",
        "generator": "benchmarks/csp/gen_xor.py",
        "fmt": "native .csp + DIMACS .cnf",
        "conversion": "load_instance (.csp or .cnf)",
        "grid": {"n": [40, 60, 80], "k": 3, "ratio": [0.5, 0.9], "seeds": 20},
        "counting_role": "random k-XOR #solutions; GF(2) rank-structured parity.",
        "seed_note": "rng = seed*100003 + n.",
    },
    {
        "id": "coloring",
        "mechanism": "low-treewidth",
        "generator": "benchmarks/csp/gen_coloring.py",
        "fmt": "DIMACS .cnf",
        "conversion": "load_instance (.cnf)",
        "grid": {"n": [40, 60, 80], "k": 4, "w": 4, "seeds": 20,
                 "families": ["banded", "random"]},
        "counting_role": "#colorings = chromatic-polynomial evaluation; treewidth "
                         "axis; overlaps GenericTensorNetworks' native problems.",
        "seed_note": "banded=predicted win; random=matched neg control.",
    },
    {
        "id": "extcsp",
        "mechanism": "random-table",
        "generator": "benchmarks/csp/gen_extcsp.py",
        "fmt": "native .csp + DIMACS .cnf",
        "conversion": "load_instance (.csp or .cnf)",
        "grid": {"note": "see gen_extcsp.py --help for the sweep knobs"},
        "counting_role": "random-table #CSP — the only family exercising truly "
                         "arbitrary native tables.",
        "seed_note": "seeded via --seed.",
    },
    {
        "id": "qcp",
        "mechanism": "all-different",
        "generator": "benchmarks/csp/gen_qcp.py",
        "fmt": "native .csp + DIMACS .cnf",
        "conversion": "load_instance (.csp or .cnf)",
        "grid": {"note": "quasigroup completion; see gen_qcp.py --help"},
        "counting_role": "#Latin-square completions; γ-rule home turf.",
        "seed_note": "seeded via --seed.",
    },
    {
        "id": "x3c",
        "mechanism": "exact-cover",
        "generator": "benchmarks/csp/gen_x3c.py",
        "fmt": "native .csp + DIMACS .cnf",
        "conversion": "load_instance (.csp or .cnf)",
        "grid": {"note": "exact cover by 3-sets; see gen_x3c.py --help"},
        "counting_role": "#exact-covers; home family for exact-cover γ branching.",
        "seed_note": "seeded via --seed.",
    },
    {
        "id": "ising",
        "mechanism": "spin-glass-partition-function",
        "generator": "benchmarks/gen_ising.py",
        "fmt": "wcn-1 JSON (native weighted tensors) — emitted directly",
        "conversion": "none (native wcn-1)",
        "grid": {"L": [3, 4, 5, 6], "topology": ["grid2d", "random3reg"],
                 "beta": "generator default", "seeds": 10},
        "counting_role": "Ising Z on native structure vs TensorOrder/GTN; small "
                         "sizes carry exact brute-force ground truth.",
        "seed_note": "brute-force exact Z for L<=5 as ground truth (meta.known_Z).",
    },
]


def all_family_ids():
    return [f["id"] for f in EXTERNAL] + [f["id"] for f in GENERATOR]
