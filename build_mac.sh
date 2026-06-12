#!/bin/bash
# build_mac.sh — Build, sign, package OnSight for distribution
set -e  

APP_NAME="OnSightPathology_App"
VERSION="1.0.0"
DMG_NAME="OnSight-${VERSION}"

echo ""
echo "========================================"
echo " OnSight Mac Build Pipeline"
echo "========================================"
echo ""

# ----- 1. clean -----
echo "[1/6] Cleaning previous build artifacts..."
rm -rf build dist

# ----- 2. PyInstaller build -----
echo "[2/6] Running PyInstaller..."
ONSIGHT_BUILD=local pyinstaller --noconfirm app_mac.spec

if [ ! -d "dist/${APP_NAME}.app" ]; then
    echo "❌ Build failed — dist/${APP_NAME}.app not found"
    exit 1
fi

# ----- 3. Signing inner binaries -----
echo "[3/6] Signing inner binaries..."
find "dist/${APP_NAME}.app" \( -name "*.dylib" -o -name "*.so" \) -print0 | \
while IFS= read -r -d '' lib; do
    codesign --force --sign - --timestamp=none "$lib" 2>/dev/null || true
done

# ----- 4. Ad-hoc signing the app bundle -----
echo "[4/6] Ad-hoc signing the app bundle..."
codesign --force --deep --sign - \
    --options runtime \
    --entitlements onsight.entitlements \
    --timestamp=none \
    "dist/${APP_NAME}.app"

# Verify the signature
codesign --verify --deep --strict --verbose=2 "dist/${APP_NAME}.app" 2>&1 | \
    tail -5

# ----- 5. Creating DMG -----
echo "[5/6] Creating DMG..."
rm -f "dist/${DMG_NAME}.dmg"

create-dmg \
    --volname "OnSight Installer" \
    --window-pos 200 120 \
    --window-size 600 400 \
    --icon-size 100 \
    --icon "${APP_NAME}.app" 175 190 \
    --hide-extension "${APP_NAME}.app" \
    --app-drop-link 425 190 \
    --no-internet-enable \
    "dist/${DMG_NAME}.dmg" \
    "dist/${APP_NAME}.app" \
    || echo "(create-dmg returned non-zero, but DMG may still be usable)"

# ----- 6. Signing the DMG itself -----
echo "[6/6] Signing the DMG..."
codesign --force --sign - "dist/${DMG_NAME}.dmg"

# ----- Summary -----
echo ""
echo "========================================"
echo " ✅ Build complete!"
echo "========================================"
echo ""
echo "  App:  dist/${APP_NAME}.app"
echo "  DMG:  dist/${DMG_NAME}.dmg"
echo ""
echo "  DMG size: $(du -h "dist/${DMG_NAME}.dmg" | cut -f1)"
echo ""
echo "Next steps:"
echo "  1. Test locally: open dist/${DMG_NAME}.dmg"
echo "  2. Upload to Drive for distribution"
echo "  3. Share download link with users"
echo ""
