#!/usr/bin/env bash
# Installs Docker + NVIDIA Container Toolkit and adds you to the docker group.
# Needs sudo. Assumes the NVIDIA driver is already installed (nvidia-smi works).
#   Arch / Omarchy / Manjaro: tested.
#   Ubuntu / Debian: follows NVIDIA's documented apt flow; not tested on this exact box.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

command -v nvidia-smi >/dev/null || die "Install the NVIDIA driver first (nvidia-open on Blackwell)."

if command -v pacman >/dev/null; then
  log "Arch-based distro"
  sudo pacman -S --needed --noconfirm docker nvidia-container-toolkit
elif command -v apt-get >/dev/null; then
  log "Debian/Ubuntu"
  if ! command -v docker >/dev/null; then
    sudo apt-get update
    sudo apt-get install -y docker.io
  fi
  if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
    sudo apt-get install -y curl gnupg
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
  fi
else
  die "Unsupported package manager. Install docker + nvidia-container-toolkit manually, then run scripts/doctor.sh."
fi

log "Registering the nvidia runtime with docker"
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl enable --now docker.service
sudo systemctl restart docker.service

if ! id -nG "$USER" | grep -qw docker; then
  sudo usermod -aG docker "$USER"
  warn "Added $USER to the docker group. Log out and back in (or open a new SSH session) before continuing."
fi

log "GPU passthrough smoke test"
sudo docker run --rm --gpus all ubuntu:24.04 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
  && ok "containers can see the GPU"
