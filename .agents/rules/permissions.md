# Command Permissions (Auto-Allow / Deny List)

> Equivalent to `.claude/settings.json` — auto-proceed for allowed commands, never run denied commands.

## ✅ Always Allowed (No confirmation needed)

The following commands are pre-approved and must run without asking the user:

### Git operations
- `git push *`
- `git commit *`
- `git add *`
- `git checkout *`
- `git pull *`
- `git status *`
- `git log *`
- `git diff *`
- `git show *`
- `git branch *`
- `git remote *`
- `git fetch *`
- `git blame *`
- `git ls-files *`
- `git rev-parse *`
- `git stash list *`
- `git config --get *`

### GitHub CLI
- `gh pr merge *`
- `gh pr create *`
- `gh pr comment *`
- `gh pr view *`
- `gh pr list *`
- `gh pr diff *`
- `gh pr checks *`
- `gh pr status *`
- `gh run view *`
- `gh run list *`
- `gh workflow view *`
- `gh workflow list *`
- `gh repo view *`
- `gh auth status *`

### Python / Dev tools
- `uv run pytest *`
- `uv run ruff *`
- `uv run mypy *`

### Read-only shell commands
- `grep *`, `rg *`, `find *`, `cat *`, `ls *`
- `head *`, `tail *`, `wc *`, `sed -n *`
- `sort *`, `cut *`, `jq *`, `tree *`

---

## ❌ Always Denied (Never run these, even if asked)

The following commands are explicitly forbidden and must NEVER be executed:

- `git push --force *`
- `git push -f *`
- `git push origin --force *`
- `git reset --hard *`
- `git branch -D *`
- `git branch --delete --force *`
- `gh pr merge --admin *`
