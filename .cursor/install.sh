#!/usr/bin/env bash
# Idempotent Cloud Agent setup for the ForecastGlioma R pipelines.
# Installs R and the CRAN packages both RF scripts need, using r2u binary
# packages for Ubuntu so setup is fast and deterministic. Safe to re-run.
set -euo pipefail

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

export DEBIAN_FRONTEND=noninteractive

# --- Base tooling ---
$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends \
  ca-certificates curl wget gnupg dirmngr locales

# --- CRAN apt repo (latest R for Ubuntu noble) ---
$SUDO install -d -m 0755 /etc/apt/trusted.gpg.d
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
  | $SUDO tee /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc >/dev/null
echo "deb [arch=amd64] https://cloud.r-project.org/bin/linux/ubuntu noble-cran40/" \
  | $SUDO tee /etc/apt/sources.list.d/cran_r.list >/dev/null

# --- r2u repo (binary CRAN packages) + pin ---
wget -qO- https://eddelbuettel.github.io/r2u/assets/dirk_eddelbuettel_key.asc \
  | $SUDO tee /etc/apt/trusted.gpg.d/cranapt_key.asc >/dev/null
echo "deb [arch=amd64] https://r2u.stat.illinois.edu/ubuntu noble main" \
  | $SUDO tee /etc/apt/sources.list.d/cranapt.list >/dev/null
printf 'Package: *\nPin: release o=CRAN-Apt Project\nPin: release l=CRAN-Apt Packages\nPin-Priority: 700\n' \
  | $SUDO tee /etc/apt/preferences.d/99cranapt >/dev/null

# --- R + pipeline packages (binaries pull their own system deps) ---
$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends \
  r-base-core r-base-dev \
  r-cran-caret \
  r-cran-proc \
  r-cran-dplyr \
  r-cran-readxl \
  r-cran-ggplot2 \
  r-cran-viridis \
  r-cran-randomforest \
  r-cran-e1071

# --- Verify the toolchain and prepare the output directory ---
mkdir -p results
Rscript -e 'pkgs <- c("caret","pROC","dplyr","readxl","ggplot2","viridis","randomForest","e1071"); invisible(lapply(pkgs, function(p) library(p, character.only = TRUE))); cat("R", as.character(getRversion()), "- all pipeline packages loaded OK\n")'
