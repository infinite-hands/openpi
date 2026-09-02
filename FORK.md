# Infinite Hands OpenPI fork

`main` is a fast-forward-only mirror of `Physical-Intelligence/openpi:main`.
Do not add Infinite Hands changes to that branch.

`ih-attention-overlay` contains the opt-in Gemma attention API used by the
Infinite Hands live diagnostic. Infinite Hands pins an exact commit from that
branch rather than a moving branch name.

The scheduled `Sync upstream OpenPI` workflow runs daily and can be started
manually. It updates `main` only when the change is a fast-forward; divergence
fails the job instead of overwriting fork-owned history.
