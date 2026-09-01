#!/usr/bin/env bash
set -euo pipefail
cd /opt/pip-guardiola

git fetch origin
branch="$(git rev-parse --abbrev-ref HEAD)"
remote_ref="origin/${branch}"

# A first-boot copy of this script may sit untracked on the box and block
# git pull. Drop untracked files that the incoming revision wants to add.
if git rev-parse --verify "$remote_ref" >/dev/null 2>&1; then
  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    if [[ -e "$path" ]] && ! git ls-files --error-unmatch "$path" >/dev/null 2>&1; then
      rm -f "$path"
    fi
  done < <(git diff --name-only "HEAD...${remote_ref}")
fi

git pull --ff-only

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
