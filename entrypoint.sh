#!/bin/bash
set -e

# Fix git ownership issue (needed each time due to volume mounts)
git config --global --add safe.directory /app

# Execute the command passed to the container (e.g., bash).
exec "$@"