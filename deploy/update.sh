#!/usr/bin/env bash
# Rebuild pipguardiola.com from origin/main.
# Safe to run as ubuntu or via sudo — git always runs as the repo owner so
# .git/FETCH_HEAD does not end up root-owned.
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/pip-guardiola}"
cd "$APP_DIR"

owner="$(stat -c '%U' "$APP_DIR")"
group="$(stat -c '%G' "$APP_DIR")"

if [[ ! -w .git || ( -e .git/FETCH_HEAD && ! -w .git/FETCH_HEAD ) ]]; then
  sudo chown -R "${owner}:${group}" "$APP_DIR"
fi

as_owner() {
  if [[ "$(id -un)" == "$owner" ]]; then
    "$@"
  else
    sudo -u "$owner" -- "$@"
  fi
}

as_owner git fetch origin
branch="$(as_owner git rev-parse --abbrev-ref HEAD)"
remote_ref="origin/${branch}"

# Untracked first-boot copies (e.g. a hand-written deploy/update.sh) block ff-only.
if as_owner git rev-parse --verify "$remote_ref" >/dev/null 2>&1; then
  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    if [[ -e "$path" ]] && ! as_owner git ls-files --error-unmatch "$path" >/dev/null 2>&1; then
      as_owner rm -f "$path"
    fi
  done < <(as_owner git diff --name-only "HEAD...${remote_ref}")
fi

as_owner git pull --ff-only

compose() {
  if docker info >/dev/null 2>&1; then
    docker compose "$@"
  else
    sudo docker compose "$@"
  fi
}

compose -f docker-compose.yml -f docker-compose.lightsail.yml up -d --build
if docker info >/dev/null 2>&1; then
  docker image prune -f
else
  sudo docker image prune -f
fi
