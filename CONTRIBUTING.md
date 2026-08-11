# Contributing

Issues and pull requests are welcome for bug fixes, documentation improvements, reproducibility fixes, and clearly scoped extensions.

Before submitting a change:

- keep REALDISP data, participant-derived data, checkpoints, generated datasets, credentials, and other sensitive material out of the repository;
- keep changes portable and avoid machine-specific paths;
- add or update tests when behavior changes;
- preserve the documented 6-channel and separately trained 3-channel model configurations unless a change is explicitly proposing a new experiment;
- do not overstate reproducibility or scientific conclusions beyond the evidence in the paper and repository.

Run the relevant tests before opening a pull request:

```bash
python -m pytest
```

For larger scientific or behavioral changes, briefly describe the motivation, affected configuration, and validation performed.
