#!/usr/bin/env bash
# First boot on an Ubuntu Lightsail instance. Run as ubuntu (sudo inside).
#   curl -fsSL https://raw.githubusercontent.com/TumeloN1/pip-guardiola/main/deploy/lightsail-setup.sh | bash
# or, after cloning:  bash deploy/lightsail-setup.sh
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/TumeloN1/pip-guardiola.git}"
APP_DIR="${APP_DIR:-/opt/pip-guardiola}"

if [[ "$(id -u)" -eq 0 ]]; then
  echo "run this as ubuntu, not root (the script sudo's where needed)"
  exit 1
fi

echo "==> swap (Lightsail 2 GB boxes OOM during the JAX image build without it)"
if ! sudo swapon --show | grep -q .; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

echo "==> docker"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
fi

echo "==> clone ${REPO_URL} → ${APP_DIR}"
sudo mkdir -p "$(dirname "$APP_DIR")"
if [[ ! -d "$APP_DIR/.git" ]]; then
  sudo git clone "$REPO_URL" "$APP_DIR"
  sudo chown -R "$USER:$USER" "$APP_DIR"
else
  sudo chown -R "$USER:$USER" "$APP_DIR"
  git -C "$APP_DIR" pull --ff-only
fi

cd "$APP_DIR"
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "wrote .env (SITE_ADDRESS=:80). Edit it if you have a domain."
fi

echo "==> build and start"
# New login group membership isn't active in this shell yet; use sudo docker.
sudo docker compose -f docker-compose.yml -f docker-compose.lightsail.yml up -d --build

echo
echo "Pip Guardiola is up."
echo "  Lightsail networking: allow SSH (22), HTTP (80), HTTPS (443). Do not open 8317."
echo "  Attach a static IP, then open http://<that-ip>/"
echo "  Domain later: set SITE_ADDRESS=your.domain in ${APP_DIR}/.env, DNS A → static IP, then:"
echo "    cd ${APP_DIR} && sudo docker compose -f docker-compose.yml -f docker-compose.lightsail.yml up -d"
echo "  Log out and back in so 'docker' works without sudo."
