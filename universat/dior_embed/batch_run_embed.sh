#! /usr/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")" && pwd > /dev/null

CONDA_ENV="openmm-10m-hypernet"

GPU_INDEX="0"

configfiles=(
    # "self-olmoearth-base-2m_1xb8-50e_m-cashew-plant-s2-linear"
    # "self-olmoearth-base-10m_1xb8-50e_m-cashew-plant-s2-linear"
    # "self-olmoearth-base-10m-8400-ema_1xb8-50e_m-cashew-plant-s2-linear"
    # "self-olmoearth-base-10m-bak_1xb8-50e_m-cashew-plant-s2-linear"
    "self-olmoearth-base-10m-hyper_1xb8-50e_m-sa-crop-type-s2-linear"
)

for configfile in "${configfiles[@]}"; do
    bash "./embed.sh" "$CONDA_ENV" "$GPU_INDEX" "$configfile"
done