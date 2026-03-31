#!/usr/bin/env bash
# LichtFeld-Studio Linux ビルドスクリプト
# 前提: Ubuntu 24.04, NVIDIA GPU, CUDA Toolkit 12.8+
#
# 使い方:
#   chmod +x scripts/build_lichtfeld_linux.sh
#   ./scripts/build_lichtfeld_linux.sh
#
# CUDA パスを指定する場合:
#   CUDA_ROOT=/usr/local/cuda-13.0 ./scripts/build_lichtfeld_linux.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LICHTFELD_DIR="$PROJECT_ROOT/LichtFeld-Studio"
VCPKG_DIR="$PROJECT_ROOT/vcpkg"

# --------------------------------------------------------------------------
# 1. システム依存パッケージ (要 sudo)
# --------------------------------------------------------------------------
install_system_deps() {
    echo "=== システム依存パッケージのインストール ==="
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
# 2. CMake 4.x インストール (3.30+ 必須)
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
            echo "=== CMake $ver は要件を満たしています。スキップ ==="
            return
        fi
    fi

    echo "=== CMake 4.0.3 のインストール ==="
    local arch
    arch=$(uname -m)
    local tmp
    tmp=$(mktemp -d)
    wget -q -O "$tmp/cmake.sh" \
        "https://github.com/Kitware/CMake/releases/download/v4.0.3/cmake-4.0.3-linux-${arch}.sh"
    chmod +x "$tmp/cmake.sh"
    sudo "$tmp/cmake.sh" --skip-license --prefix=/usr/local
    rm -rf "$tmp"
    echo "CMake $(cmake --version | head -1) インストール完了"
}

# --------------------------------------------------------------------------
# 3. vcpkg セットアップ
# --------------------------------------------------------------------------
setup_vcpkg() {
    if [ -x "$VCPKG_DIR/vcpkg" ]; then
        echo "=== vcpkg は既にセットアップ済み。スキップ ==="
    else
        echo "=== vcpkg のクローンとブートストラップ ==="
        git clone https://github.com/microsoft/vcpkg.git "$VCPKG_DIR"
        "$VCPKG_DIR/bootstrap-vcpkg.sh" -disableMetrics
    fi
    export VCPKG_ROOT="$VCPKG_DIR"
}

# --------------------------------------------------------------------------
# 4. CUDA パスの検出
# --------------------------------------------------------------------------
detect_cuda() {
    if [ -n "${CUDA_ROOT:-}" ] && [ -x "$CUDA_ROOT/bin/nvcc" ]; then
        echo "=== CUDA_ROOT から使用: $CUDA_ROOT ==="
        return
    fi

    # /usr/local/cuda-* から最新バージョンを検出
    local latest=""
    for d in /usr/local/cuda-*/bin/nvcc; do
        [ -x "$d" ] && latest="$(dirname "$(dirname "$d")")"
    done

    if [ -n "$latest" ]; then
        CUDA_ROOT="$latest"
        echo "=== 自動検出 CUDA: $CUDA_ROOT ==="
    elif [ -x /usr/local/cuda/bin/nvcc ]; then
        CUDA_ROOT="/usr/local/cuda"
        echo "=== デフォルト CUDA: $CUDA_ROOT ==="
    else
        echo "エラー: CUDA Toolkit が見つかりません。CUDA_ROOT を設定してください。" >&2
        exit 1
    fi

    # バージョンチェック (12.8+ 必須)
    local cuda_ver
    cuda_ver=$("$CUDA_ROOT/bin/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    local cuda_major cuda_minor
    cuda_major=$(echo "$cuda_ver" | cut -d. -f1)
    cuda_minor=$(echo "$cuda_ver" | cut -d. -f2)
    if (( cuda_major < 12 || (cuda_major == 12 && cuda_minor < 8) )); then
        echo "エラー: CUDA $cuda_ver が検出されましたが、12.8+ が必要です。" >&2
        exit 1
    fi
    echo "=== CUDA $cuda_ver ($CUDA_ROOT) ==="
}

# --------------------------------------------------------------------------
# 5. LichtFeld-Studio サブモジュール初期化 & ビルド
# --------------------------------------------------------------------------
build_lichtfeld() {
    echo "=== サブモジュールの初期化 ==="
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

    echo "=== ビルド ($(nproc) コア) ==="
    cmake --build build -- -j"$(nproc)"

    echo "=== ビルド完了 ==="
    "$LICHTFELD_DIR/build/LichtFeld-Studio" --version
}

# --------------------------------------------------------------------------
# メイン
# --------------------------------------------------------------------------
main() {
    echo "============================================"
    echo " LichtFeld-Studio Linux ビルド"
    echo "============================================"

    install_system_deps
    install_cmake
    setup_vcpkg
    detect_cuda
    build_lichtfeld

    echo ""
    echo "============================================"
    echo " 完了!"
    echo " バイナリ: $LICHTFELD_DIR/build/LichtFeld-Studio"
    echo "============================================"
}

main "$@"
