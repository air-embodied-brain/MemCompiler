#!/bin/bash
# Run Manager-Executor architecture on ALFWorld
# Usage: bash run_alfworld.sh

# ---- Environment Variables ----
export MEMCOMPILER_TMPDIR="/tmp/memcompiler"

# OpenAI API (used when API_TYPE="openai" and MANAGER_MODEL="gpt")
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export OPENAI_BASE_URL="https://api.openai.com/v1"

# Azure OpenAI (used when API_TYPE="azure")
export AZURE_OPENAI_API_KEY="YOUR_AZURE_API_KEY"
export AZURE_OPENAI_ENDPOINT="https://YOUR_ENDPOINT.openai.azure.com/"

# Gemini (used when MANAGER_MODEL="gemini")
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
export GOOGLE_GEMINI_BASE_URL="https://generativelanguage.googleapis.com"

# Optional: control Gemini thinking
# export MEMCOMPILER_GEMINI_DISABLE_THINKING=1
# export MEMCOMPILER_GEMINI_THINKING_LEVEL=minimal

# ---- Run ----
python tasks/run_manager_executor.py
