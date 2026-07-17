#!/bin/bash
# Script to easily update Declearn version before a release.
# 
# Update the version in the same way in all concerned files.
# Usage: ./update_version.sh CURRENT_VERSION NEW_VERSION

# Set bash strict mode.
set -euo pipefail
IFS=$'\n\t'

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 CURRENT_VERSION NEW_VERSION"
    exit 1
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PYPROJECT_PATH="$SCRIPT_DIR/../pyproject.toml"
VERSION_PY_PATH="$SCRIPT_DIR/../declearn/version.py"

old_version=$1
new_version=$2

# Replace version in pyproject.toml.
old_string="version = \"$old_version\""
new_string="version = \"$new_version\""
sed -i "0,/$old_string/s/$old_string/$new_string/" "$PYPROJECT_PATH"
# Note: for robustness, we use a sed pattern to only replace the first
# occurrence of the first matching line.
echo "Version updated in 'pyproject.toml'."

# Replace version in version.py (only the first occurence)
old_string="VERSION = \"$old_version\""
new_string="VERSION = \"$new_version\""
sed -i "0,/$old_string/s/$old_string/$new_string/" "$VERSION_PY_PATH"
echo "Version updated in 'version.py'."