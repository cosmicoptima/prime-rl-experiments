curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv sync && uv sync --all-extras
uv pip uninstall flash_attn
uv pip install flash_attn --no-build-isolation --no-cache-dir

# Prime-RL setup for runpod persistence

# Create symlink to persistent uv installation
ln -sf /workspace/.local ~/.local
# Add uv to PATH
export PATH="/workspace/.local/bin:$PATH"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TMPDIR=/workspace/tmp

git config --global credential.helper store

uv run hf auth login
uv run wandb login