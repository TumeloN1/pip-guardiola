#!/usr/bin/env bash
set -euo pipefail
cd /opt/pip-guardiola
git pull --ff-only
if docker info >/dev/null 2>&1; then
  docker compose -f docker-compose.yml -f docker-compose.lightsail.yml up -d --build
  docker image prune -f
else
  sudo docker compose -f docker-compose.yml -f docker-compose.lightsail.yml up -d --build
  sudo docker image prune -f
fi
