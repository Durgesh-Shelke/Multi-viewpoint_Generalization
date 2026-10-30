#!/usr/bin/env bash
# scripts/export_frozen_code.sh
# Exports frozen code at the specified git tag (default: pre-pilot-freeze)
# into a read-only directory frozen_code/<tag>/ for reproducible execution.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EXP_DIR"

TAG="${1:-v2-freeze}"
TARGET_DIR="$EXP_DIR/frozen_code/$TAG"

echo "=============================================================================="
echo " EXPORTING FROZEN CODE"
echo "=============================================================================="
echo " Tag:         $TAG"
echo " Target Dir:  $TARGET_DIR"

# 1. Verify that tag exists in git
if ! git rev-parse -q --verify "$TAG" >/dev/null; then
    echo "Error: Git tag '$TAG' does not exist in repository $EXP_DIR" >&2
    exit 1
fi

COMMIT_HASH="$(git rev-parse "$TAG")"
echo " Commit Hash: $COMMIT_HASH"

# 2. Clean up previous target directory if it exists
if [ -d "$TARGET_DIR" ]; then
    echo " Removing previous export at $TARGET_DIR..."
    chmod -R u+w "$TARGET_DIR" 2>/dev/null || true
    rm -rf "$TARGET_DIR"
fi

mkdir -p "$TARGET_DIR"

# 3. Export archive from git tag into target directory
echo " Extracting archive from tag '$TAG'..."
git archive "$TAG" | tar -x -C "$TARGET_DIR"

# 4. Record commit hash metadata files inside frozen export
echo "$COMMIT_HASH" > "$TARGET_DIR/COMMIT_HASH"
echo "$COMMIT_HASH" > "$TARGET_DIR/.git_commit"

# 5. Lock all files and directories as read-only
echo " Setting read-only permissions (chmod -R a-w)..."
chmod -R a-w "$TARGET_DIR"

echo "------------------------------------------------------------------------------"
echo " Frozen export successfully created at: $TARGET_DIR"
echo " Exported commit: $COMMIT_HASH"
echo " Directory status: READ-ONLY"
echo "=============================================================================="
