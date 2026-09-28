#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: bash scripts/run_docker.sh scripts/rsl_rl/train.py --task=G1-flat-vel --headless [options]" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker_network_args=()
docker_env_args=()
docker_env_file_args=()
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[i]}" == "--livestream=1" || "${args[i]}" == "--livestream=2" ]] ||
        { [[ "${args[i]}" == "--livestream" ]] &&
          (( i + 1 < ${#args[@]} )) &&
          [[ "${args[i + 1]}" == "1" || "${args[i + 1]}" == "2" ]]; }; then
        # WebRTC advertises the host's address; Docker bridge networking breaks remote connections.
        docker_network_args=(--network=host)
        break
    fi
done

# Load a persistent W&B credential file when present. Keep this file outside the repository.
user_home_dir="$(getent passwd "$(id -u)" | cut -d: -f6)"
wandb_env_file="${ROBOT_RL_WANDB_ENV_FILE:-${user_home_dir}/.config/robot_rl/wandb.env}"
if [[ -f "${wandb_env_file}" ]]; then
    docker_env_file_args=(--env-file "${wandb_env_file}")
fi

# Forward W&B settings without baking credentials into the image or repository.
for env_name in WANDB_API_KEY WANDB_ENTITY WANDB_BASE_URL WANDB_MODE; do
    if [[ -n "${!env_name:-}" ]]; then
        docker_env_args+=(--env "${env_name}")
    fi
done

exec docker run --rm "${docker_network_args[@]}" "${docker_env_file_args[@]}" "${docker_env_args[@]}" \
    --runtime=nvidia --gpus all --shm-size=4g \
    --user "$(id -u):$(id -g)" --group-add 1234 \
    --env HOME=/tmp --env PYTHONPATH=/workspace/robot_rl/source/robot_rl \
    --volume "${repo_root}:/workspace/robot_rl" \
    --workdir /workspace/robot_rl \
    --entrypoint /isaac-sim/python.sh \
    robot-rl:local "$@"
