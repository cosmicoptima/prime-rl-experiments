curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv sync && uv sync --all-extras

# Prime-RL setup for runpod persistence

# Create symlink to persistent uv installation
ln -sf /workspace/.local ~/.local
# Add uv to PATH
export PATH="/workspace/.local/bin:$PATH"