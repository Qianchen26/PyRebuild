"""
Synthetic scenario generation, save, and load.

A "scenario" is a frozen list of buildings (uid, dl, pl, total_work,
max_crew, arrive_day) plus the contractor count.  Saving scenarios to
disk ensures exact reproducibility across training runs without paying
the generation cost again.

Usage:
    # Generate and save
    python -m rl.scenarios --n 20 --tag balanced

    # In train.py: load from disk
    from code.scenarios import load_scenario
    cfg = load_scenario("output/scenarios/balanced_seed0.json")
    env = HouseRecoveryEnv.from_scenario(cfg)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as C

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_DIR = PROJECT_ROOT / "output" / "scenarios"


# ──────────────────────────────────────────────────────────────────────
# Data structure
# ──────────────────────────────────────────────────────────────────────

@dataclass
class BuildingSpec:
    """Frozen building specification (no mutable simulation state)."""
    uid: int
    dl: int
    pl: int
    total_work: float
    max_crew: int
    arrive_day: int


@dataclass
class ScenarioConfig:
    """Everything needed to reconstruct an environment identically."""
    name: str                        # e.g. "balanced_seed0"
    seed: int
    n_buildings: int
    n_contractors: int
    damage_dist: Tuple[float, float, float]
    infra_ratio: float
    max_crew: List[int]              # [2, 4, 6]
    batch_schedule: List[Tuple[int, float]]
    buildings: List[BuildingSpec]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "n_buildings": self.n_buildings,
            "n_contractors": self.n_contractors,
            "damage_dist": list(self.damage_dist),
            "infra_ratio": self.infra_ratio,
            "max_crew": list(self.max_crew),
            "batch_schedule": [list(x) for x in self.batch_schedule],
            "buildings": [asdict(b) for b in self.buildings],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioConfig":
        buildings = [BuildingSpec(**b) for b in d["buildings"]]
        return cls(
            name=d["name"],
            seed=d["seed"],
            n_buildings=d["n_buildings"],
            n_contractors=d["n_contractors"],
            damage_dist=tuple(d["damage_dist"]),
            infra_ratio=d["infra_ratio"],
            max_crew=d["max_crew"],
            batch_schedule=[tuple(x) for x in d["batch_schedule"]],
            buildings=buildings,
        )


# ──────────────────────────────────────────────────────────────────────
# Generate
# ──────────────────────────────────────────────────────────────────────

def generate_scenario(
    seed: int,
    scenario_type: str = "balanced",
    tag: Optional[str] = None,
) -> ScenarioConfig:
    """Generate a synthetic scenario by creating a temporary env and
    extracting the frozen building list."""
    # Lazy import to avoid circular dependency
    from .env import HouseRecoveryEnv

    damage_dist = C.SYNTH_DAMAGE_DISTS.get(
        scenario_type, C.SYNTH_DAMAGE_DISTS["balanced"]
    )
    rng = np.random.default_rng(seed)
    n_buildings = int(rng.integers(*C.SYNTH_BUILDINGS_RANGE))
    contractor_ratio = float(rng.uniform(*C.SYNTH_CONTRACTOR_RATIO_RANGE))
    n_contractors = max(1, int(round(n_buildings * contractor_ratio)))

    infra_ratio = float(rng.uniform(*C.SYNTH_INFRA_RATIO_RANGE))

    # Create env and reset to generate buildings
    env = HouseRecoveryEnv(
        n_buildings=n_buildings,
        n_contractors=n_contractors,
        damage_dist=damage_dist,
        infra_ratio=infra_ratio,
    )
    env.reset(seed=seed)

    # Freeze buildings
    specs = [
        BuildingSpec(
            uid=b.uid, dl=b.dl, pl=b.pl,
            total_work=b.total_work,
            max_crew=b.max_crew,
            arrive_day=b.arrive_day,
        )
        for b in env.buildings
    ]

    name = f"{tag or scenario_type}_seed{seed}"
    return ScenarioConfig(
        name=name,
        seed=seed,
        n_buildings=n_buildings,
        n_contractors=n_contractors,
        damage_dist=tuple(damage_dist),
        infra_ratio=infra_ratio,
        max_crew=C.MAX_CREW.tolist(),
        batch_schedule=list(C.BATCH_SCHEDULE),
        buildings=specs,
    )


# ──────────────────────────────────────────────────────────────────────
# Save / Load
# ──────────────────────────────────────────────────────────────────────

def save_scenario(cfg: ScenarioConfig, directory: Optional[Path] = None) -> Path:
    """Save scenario to JSON.  Returns the file path."""
    d = directory or SCENARIO_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{cfg.name}.json"
    with open(path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    return path


def load_scenario(path: str | Path) -> ScenarioConfig:
    """Load scenario from JSON file."""
    with open(path) as f:
        return ScenarioConfig.from_dict(json.load(f))


def list_scenarios(directory: Optional[Path] = None) -> List[Path]:
    """List all saved scenario files."""
    d = directory or SCENARIO_DIR
    if not d.exists():
        return []
    return sorted(d.glob("*.json"))


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and save synthetic scenarios"
    )
    parser.add_argument("--n", type=int, default=5, help="Number of scenarios")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument(
        "--type", type=str, default="balanced",
        choices=list(C.SYNTH_DAMAGE_DISTS.keys()),
    )
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--list", action="store_true", help="List saved scenarios")
    args = parser.parse_args()

    if args.list:
        paths = list_scenarios()
        if not paths:
            print("No saved scenarios found.")
        for p in paths:
            cfg = load_scenario(p)
            print(
                f"  {p.name:40s}  "
                f"buildings={cfg.n_buildings:6d}  "
                f"contractors={cfg.n_contractors:6d}  "
                f"dist={cfg.damage_dist}"
            )
        return

    print(f"Generating {args.n} scenarios (type={args.type})...")
    for i in range(args.n):
        seed = args.start_seed + i
        cfg = generate_scenario(seed, args.type, args.tag)
        path = save_scenario(cfg)
        print(
            f"  [{i+1}/{args.n}] {path.name}  "
            f"buildings={cfg.n_buildings}  contractors={cfg.n_contractors}"
        )
    print(f"Saved to {SCENARIO_DIR}/")


if __name__ == "__main__":
    main()
