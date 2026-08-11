# Milestone 5 handoff

## Scope

Milestone 5 adds release engineering only. Scientific preprocessing, VAE,
Rectified Flow, generation, evaluation, and analysis implementations are
unchanged.

## Packaging

- Python floor: 3.10.
- Core dependencies: PyYAML>=6.0 and NumPy>=1.21.3.
- Optional extras: test, training, evaluation, analysis, and dev.
- Build backend: setuptools>=69 with wheel.
- Exact author, title, journal, and DOI metadata are regression tested.
- Publication metadata was validated without changing scientific behavior.

The disposable Python 3.11 build produced:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| lrf-imu-0.1.0.tar.gz | 167120 | 802ac9e667564a2fc7906c88d3fbdcae93a5c159c909bb93fe3a45c0cb09e0b6 |
| lrf_imu-0.1.0-py3-none-any.whl | 147481 | a4038290ab7e3d6e538308dfe82eb00b0dc4f8dea8f8c422d291ba9d3c2f49ac |

Both archives contained the three packaged paper configurations and no
prohibited payload suffixes. The artifacts were validation outputs only and
were not committed.

## Installation and command validation

Separate clean Python 3.11 environments installed the wheel and source
distribution. Imports resolved only from each environment's site-packages.
From a foreign working directory, both installs passed:

- package import;
- console and module version commands;
- prepare-data, vae-smoke, flow-smoke, evaluate, and reproduce-core help;
- synthetic prepare-data with 16 training, 7 VAE-validation, and 8 test windows.

A clean Python 3.10 environment installed and compiled the wheel and passed
import/version checks. A separate installed-wheel CPU runtime passed 6CH and
3CH VAE smokes and the default width-256 Flow smoke. This runtime reused a
locally validated CPU PyTorch installation; it was not a clean dependency
resolution test.

## Tests and safety

- Packaging and safety focus: 14 passed.
- Complete source-layout pytest: 169 passed, 1 optional skip in 23.38 seconds.
- Ruff and mypy: passed.
- TOML, workflow YAML, and citation parsing: passed.
- Physical files above 5 MiB: zero.
- Git object store: 877.13 KiB, no pack or garbage.
- No checkpoint, generated array, participant data, Results payload, secret,
  build product, or cache is part of the intended Git delta.
- CI covers Python 3.10 and 3.12 core tests, wheel/sdist build, foreign-CWD wheel
  import/help smoke, and the repository safety scanner. GitHub Actions was not
  executed remotely.

## Later publication decision

The later code-only publication decision supersedes this milestone's
provisional release posture. It does not change the exclusion of REALDISP,
checkpoints, generated datasets, trained models, historical Results payloads,
manuscript files, or figures.

No remote, tag, push, upload, or publication action was performed in Milestone 5.
