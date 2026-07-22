#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <repo-owner> <repo-name> [ssh-key-path]"
  echo "Example: $0 18004855049 Evolution-Hand ~/.ssh/evolution_hand_github"
  exit 1
fi

repo_owner="$1"
repo_name="$2"
key_path="${3:-$HOME/.ssh/evolution_hand_github}"
remote_url="git@github.com:${repo_owner}/${repo_name}.git"
ssh_config_path="$HOME/.ssh/config"

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

if [[ ! -f "${key_path}" ]]; then
  ssh-keygen -t ed25519 -C "${repo_name}-github" -f "${key_path}" -N ""
  echo "Created SSH key: ${key_path}"
fi

chmod 600 "${key_path}"
chmod 644 "${key_path}.pub"

if ! grep -q "Host github.com-evolution-hand" "${ssh_config_path}" 2>/dev/null; then
  cat >>"${ssh_config_path}" <<EOF

Host github.com-evolution-hand
  HostName github.com
  User git
  IdentityFile ${key_path}
  IdentitiesOnly yes
EOF
  chmod 600 "${ssh_config_path}"
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "git@github.com-evolution-hand:${repo_owner}/${repo_name}.git"
else
  git remote add origin "git@github.com-evolution-hand:${repo_owner}/${repo_name}.git"
fi

echo "Public key:"
cat "${key_path}.pub"
echo
echo "Add this key to GitHub SSH keys, then run:"
echo "  ssh -T git@github.com-evolution-hand"
echo "  git push -u origin main"
echo
echo "Configured origin -> ${remote_url}"
