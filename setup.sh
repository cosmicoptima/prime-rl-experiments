apt update
apt install vim

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
\. "$HOME/.nvm/nvm.sh"
nvm install 22
npm install -g @anthropic-ai/claude-code

# Prime-RL setup for runpod persistence

# Create symlink to persistent uv installation
ln -sf /workspace/.local ~/.local
# Add uv to PATH
export PATH="/workspace/.local/bin:$PATH"
# Set working directory
cd /workspace/prime-rl
