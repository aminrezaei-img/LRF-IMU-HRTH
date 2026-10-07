# Contributing

Issues and pull requests are welcome for bug fixes, documentation improvements, reproducibility fixes, and clearly scoped extensions.

Before submitting a change:

- keep raw data, checkpoints, generated datasets, credentials, and other sensitive material out of normal Git history; approved large public artifacts belong only in a checksum-locked repository release;
- keep changes portable and avoid machine-specific paths;
- add or update tests when behavior changes;
- preserve the documented 6-channel and separately trained 3-channel model configurations unless a change is explicitly proposing a new experiment;
- keep contextual mapping, cohort orchestration, stitching, and multimodal linkage outside the model-core package;
- do not overstate reproducibility or scientific conclusions beyond the evidence in the paper and repository.

Run the relevant tests before opening a pull request:

```bash
python -m pytest
```

For larger scientific or behavioral changes, briefly describe the motivation, affected configuration, and validation performed.
