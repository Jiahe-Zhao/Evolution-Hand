#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <repo-owner> <repo-name> [ssh-key-path]"
  echo "Example: $0 Jiahe-Zhao Evolution-Hand ~/.ssh/evolution_hand_github"
  exit 1
fi

repo_owner="$1"
repo_name="$2"
key_path="${3:-$HOME/.ssh/evolution_hand_github}"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

cd "$repo_root"

bash scripts/setup_github_ssh.sh "$repo_owner" "$repo_name" "$key_path"

git branch --show-current >/dev/null
git push -u origin main

echo "Pushed main to git@github.com:${repo_owner}/${repo_name}.git"
