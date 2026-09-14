# Cloud Agent base image for the ForecastGlioma R project.
# Installs R and the CRAN packages the RF pipelines need, using r2u
# (binary CRAN packages for Ubuntu) so builds are fast and deterministic.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    TZ=UTC

# Base tooling required in fully custom images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl wget gnupg dirmngr locales sudo git \
    && sed -i '/en_US.UTF-8/s/^# //' /etc/locale.gen \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# CRAN apt repo (latest R for Ubuntu noble) + r2u binary CRAN packages.
RUN install -d -m 0755 /etc/apt/trusted.gpg.d \
    && wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
        | tee /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc >/dev/null \
    && echo "deb [arch=amd64] https://cloud.r-project.org/bin/linux/ubuntu noble-cran40/" \
        > /etc/apt/sources.list.d/cran_r.list \
    && wget -qO- https://eddelbuettel.github.io/r2u/assets/dirk_eddelbuettel_key.asc \
        | tee /etc/apt/trusted.gpg.d/cranapt_key.asc >/dev/null \
    && echo "deb [arch=amd64] https://r2u.stat.illinois.edu/ubuntu noble main" \
        > /etc/apt/sources.list.d/cranapt.list \
    && printf 'Package: *\nPin: release o=CRAN-Apt Project\nPin: release l=CRAN-Apt Packages\nPin-Priority: 700\n' \
        > /etc/apt/preferences.d/99cranapt

# R plus the packages used by both RF scripts (binary installs pull their
# own system dependencies automatically via r2u).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        r-base-core r-base-dev \
        r-cran-caret \
        r-cran-proc \
        r-cran-dplyr \
        r-cran-readxl \
        r-cran-ggplot2 \
        r-cran-viridis \
        r-cran-randomforest \
        r-cran-e1071 \
    && rm -rf /var/lib/apt/lists/*

USER ubuntu
