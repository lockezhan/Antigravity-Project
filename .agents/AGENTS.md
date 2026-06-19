# Antigravity Project Agent Rules

## Git History and Artifacts Management
- **Awareness of Git Objects**: Deleting large artifacts (like `.pt` checkpoints or heavy `.log` files) from the working directory using `git rm` does NOT remove them from the repository's size payload. They remain in the `.git/objects` history, which severely impacts `git clone` and `git pull` speeds across network-constrained environments (like downloading to Linux servers).
- **Rule**: NEVER stage or commit model weight files (`*.pt`, `*.pth`) or heavy log data to Git unless explicitly required. Always rely on `.gitignore` to shield the Git history. If they accidentally enter the history, the repository must be slimmed down using `git filter-branch` or `git filter-repo`, followed by `git gc`, rather than a simple `git rm`.
