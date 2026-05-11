#!/usr/bin/env bash
# Cut a new release. Bumps the workspace version, runs the same checks
# CI runs, commits, and tags. By default stops before pushing so you can
# inspect; pass --push to send the commit + tag and let CI build the
# .deb and publish to AUR.
#
# Usage:
#   scripts/release.sh 0.1.3            # bump + commit + tag, no push
#   scripts/release.sh 0.1.3 --push     # bump + commit + tag + push
set -euo pipefail

new="${1:-}"
push_flag="${2:-}"

if [[ -z "$new" ]]; then
  echo "Usage: $0 <X.Y.Z> [--push]" >&2
  exit 1
fi
if [[ ! "$new" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "ERROR: version must be X.Y.Z, got '$new'" >&2
  exit 1
fi
if [[ -n "$push_flag" && "$push_flag" != "--push" ]]; then
  echo "ERROR: unknown flag '$push_flag' (only --push is supported)" >&2
  exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

# --- Preflight ---
branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$branch" != "main" ]]; then
  echo "ERROR: must be on main, currently on '$branch'" >&2
  exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: working tree is dirty — commit or stash first" >&2
  exit 1
fi
git fetch --quiet origin main
if [[ "$(git rev-parse @)" != "$(git rev-parse @{u})" ]]; then
  echo "ERROR: local main is out of sync with origin/main" >&2
  exit 1
fi
if git rev-parse -q --verify "refs/tags/v$new" >/dev/null; then
  echo "ERROR: tag v$new already exists" >&2
  exit 1
fi

current="$(
  awk '
    $0 == "[workspace.package]" { in_section = 1; next }
    /^\[/ && in_section { exit }
    in_section && $1 == "version" {
      gsub(/"/, "", $3)
      print $3
      exit
    }
  ' Cargo.toml
)"
if [[ -z "$current" ]]; then
  echo "ERROR: could not read current version from Cargo.toml" >&2
  exit 1
fi
if [[ "$current" == "$new" ]]; then
  echo "ERROR: new version $new matches current — nothing to bump" >&2
  exit 1
fi

echo "Release: $current → $new"
echo

# --- Bump ---
sed -i -E "0,/^version = \"$current\"$/ s||version = \"$new\"|" Cargo.toml
sed -i -E "s|(lb-pipeline = \\{ path = \"\\.\\./pipeline\", version = \")$current(\" \\})|\\1$new\\2|" crates/app/Cargo.toml
cargo update --quiet -p linux-broadcast -p lb-pipeline

# --- Verify (same gates as CI) ---
echo "=== cargo fmt --all -- --check ==="
cargo fmt --all -- --check
echo "=== cargo clippy --workspace --all-targets -- -D warnings ==="
cargo clippy --workspace --all-targets --quiet -- -D warnings
echo "=== cargo test --workspace --quiet ==="
cargo test --workspace --quiet

# --- Commit + tag ---
git add Cargo.toml Cargo.lock crates/app/Cargo.toml
git commit --quiet -m "Release $new"
git tag -a "v$new" -m "v$new"

echo
echo "Created locally:"
git log -1 --oneline
echo "  tag: v$new"
echo

if [[ "$push_flag" == "--push" ]]; then
  echo "=== pushing to origin ==="
  git push origin main
  git push origin "v$new"
  echo
  cat <<EOF
Done. CI is now:
  • Building the .deb
  • Publishing the GitHub release
  • Pushing linux-broadcast-bin $new-1 to AUR

Watch:    https://github.com/Pedrojok01/linux-broadcast/actions
Release:  https://github.com/Pedrojok01/linux-broadcast/releases/tag/v$new
AUR:      https://aur.archlinux.org/packages/linux-broadcast-bin
EOF
else
  cat <<EOF
Not pushed (no --push flag). To ship:
  git push origin main
  git push origin v$new

Or re-run with --push next time:
  scripts/release.sh $new --push
EOF
fi
