#!/usr/bin/env bash
# LichtFeld-Studio Linux build script
# Prerequisites: Ubuntu 24.04, NVIDIA GPU, CUDA Toolkit 12.8+
#
# Usage:
#   chmod +x scripts/build_lichtfeld_linux.sh
#   ./scripts/build_lichtfeld_linux.sh
#
# To specify a custom CUDA path:
#   CUDA_ROOT=/usr/local/cuda-13.0 ./scripts/build_lichtfeld_linux.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LICHTFELD_DIR="$PROJECT_ROOT/LichtFeld-Studio"
VCPKG_DIR="$PROJECT_ROOT/vcpkg"

# --------------------------------------------------------------------------
# 1. System dependencies (requires sudo)
# --------------------------------------------------------------------------
install_system_deps() {
    echo "=== Installing system dependencies ==="
    sudo apt-get update
    sudo apt-get install -y \
        gcc-14 g++-14 gfortran-14 \
        ninja-build nasm \
        autoconf autoconf-archive automake libtool \
        pkg-config zip unzip \
        libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev

    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 60
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 60
    sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-14 60
    sudo update-alternatives --set gcc /usr/bin/gcc-14
    sudo update-alternatives --set g++ /usr/bin/g++-14
    sudo update-alternatives --set gfortran /usr/bin/gfortran-14
}

# --------------------------------------------------------------------------
# 2. CMake 4.x installation (3.30+ required)
# --------------------------------------------------------------------------
install_cmake() {
    local required_major=3
    local required_minor=30

    if command -v cmake &>/dev/null; then
        local ver
        ver=$(cmake --version | head -1 | grep -oP '[0-9]+\.[0-9]+\.[0-9]+')
        local major minor
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if (( major > required_major || (major == required_major && minor >= required_minor) )); then
            echo "=== CMake $ver meets requirements. Skipping ==="
            return
        fi
    fi

    echo "=== Installing CMake 4.0.3 ==="
    local arch
    arch=$(uname -m)
    local tmp
    tmp=$(mktemp -d)
    wget -q -O "$tmp/cmake.sh" \
        "https://github.com/Kitware/CMake/releases/download/v4.0.3/cmake-4.0.3-linux-${arch}.sh"
    chmod +x "$tmp/cmake.sh"
    sudo "$tmp/cmake.sh" --skip-license --prefix=/usr/local
    rm -rf "$tmp"
    echo "CMake $(cmake --version | head -1) installed"
}

# --------------------------------------------------------------------------
# 3. vcpkg setup
# --------------------------------------------------------------------------
setup_vcpkg() {
    if [ -x "$VCPKG_DIR/vcpkg" ]; then
        echo "=== vcpkg already set up. Skipping ==="
    else
        echo "=== Cloning and bootstrapping vcpkg ==="
        git clone https://github.com/microsoft/vcpkg.git "$VCPKG_DIR"
        "$VCPKG_DIR/bootstrap-vcpkg.sh" -disableMetrics
    fi
    export VCPKG_ROOT="$VCPKG_DIR"
}

# --------------------------------------------------------------------------
# 4. CUDA detection
# --------------------------------------------------------------------------
detect_cuda() {
    if [ -n "${CUDA_ROOT:-}" ] && [ -x "$CUDA_ROOT/bin/nvcc" ]; then
        echo "=== Using CUDA from CUDA_ROOT: $CUDA_ROOT ==="
        return
    fi

    # Find the latest version under /usr/local/cuda-*
    local latest=""
    for d in /usr/local/cuda-*/bin/nvcc; do
        [ -x "$d" ] && latest="$(dirname "$(dirname "$d")")"
    done

    if [ -n "$latest" ]; then
        CUDA_ROOT="$latest"
        echo "=== Auto-detected CUDA: $CUDA_ROOT ==="
    elif [ -x /usr/local/cuda/bin/nvcc ]; then
        CUDA_ROOT="/usr/local/cuda"
        echo "=== Default CUDA: $CUDA_ROOT ==="
    else
        echo "Error: CUDA Toolkit not found. Please set CUDA_ROOT." >&2
        exit 1
    fi

    # Version check (12.8+ required)
    local cuda_ver
    cuda_ver=$("$CUDA_ROOT/bin/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    local cuda_major cuda_minor
    cuda_major=$(echo "$cuda_ver" | cut -d. -f1)
    cuda_minor=$(echo "$cuda_ver" | cut -d. -f2)
    if (( cuda_major < 12 || (cuda_major == 12 && cuda_minor < 8) )); then
        echo "Error: CUDA $cuda_ver detected, but 12.8+ is required." >&2
        exit 1
    fi
    echo "=== CUDA $cuda_ver ($CUDA_ROOT) ==="
}

# --------------------------------------------------------------------------
# 5. LichtFeld-Studio submodule init & build
# --------------------------------------------------------------------------
build_lichtfeld() {
    echo "=== Initializing submodules ==="
    cd "$PROJECT_ROOT"
    git submodule update --init --recursive

    echo "=== CMake Configure ==="
    export PATH="$CUDA_ROOT/bin:$PATH"
    cd "$LICHTFELD_DIR"
    cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -G Ninja \
        -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
        -DCUDAToolkit_ROOT="$CUDA_ROOT"

    echo "=== Building ($(nproc) cores) ==="
    cmake --build build -- -j"$(nproc)"

    echo "=== Build complete ==="
    "$LICHTFELD_DIR/build/LichtFeld-Studio" --version
}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
main() {
    echo "============================================"
    echo " LichtFeld-Studio Linux Build"
    echo "============================================"

    install_system_deps
    install_cmake
    setup_vcpkg
    detect_cuda
    build_lichtfeld

    echo ""
    echo "============================================"
    echo " Done!"
    echo " Binary: $LICHTFELD_DIR/build/LichtFeld-Studio"
    echo "============================================"
}

main "$@"
