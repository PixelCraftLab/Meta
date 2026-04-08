#!/usr/bin/env bash
set -e

echo "Starting OpenEnv deployment process..."

cd kernel_env

# Check if HF_TOKEN is set or provided as argument
if [ -z "$HF_TOKEN" ] && [ ! -f ~/.cache/huggingface/token ]; then
    echo "Hugging Face token not found."
    read -sp "Please paste your Hugging Face Access Token: " HF_TOKEN_INPUT
    echo ""
    export HF_TOKEN="$HF_TOKEN_INPUT"
    
    # Authenticate via python
    ./.venv/bin/python -c "import huggingface_hub; huggingface_hub.login(token='$HF_TOKEN_INPUT', add_to_git_credential=True)"
fi

echo "Deploying the OpenEnv environment via CLI..."
./.venv/bin/openenv push

echo "Deployment complete! Your environment should now be available on Hugging Face Spaces."
