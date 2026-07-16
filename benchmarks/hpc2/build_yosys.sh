#!/bin/bash
#SBATCH -J bi-yosys-build
#SBATCH -p debug
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -D /hpc2hdd/home/xpan432/Codes/yosys
#SBATCH -o /hpc2ssd/JH_DATA/spooler/xpan432/boolean-inference/logs/yosys-build-%j.out
#SBATCH -e /hpc2ssd/JH_DATA/spooler/xpan432/boolean-inference/logs/yosys-build-%j.err

source /usr/local/Modules/init/bash || exit 1
module load compilers/gcc-12.2.0 || exit 1

set -euo pipefail

PREFIX=/hpc2ssd/JH_DATA/spooler/xpan432/boolean-inference/build/yosys-0.66
REPO=/hpc2hdd/home/xpan432/Codes/boolean-inference
BUILD_ENV=/hpc2hdd/home/xpan432/envs/yosys-0.66-build
PYTHON="$REPO/.venv/bin/python"
SMOKE=/hpc2ssd/JH_DATA/spooler/xpan432/boolean-inference/data/setup-smoke

for header in tcl.h readline/readline.h zlib.h ffi.h; do
  test -f "$BUILD_ENV/include/$header"
done

export PATH="$BUILD_ENV/bin:$PATH"
export CPATH="$BUILD_ENV/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$BUILD_ENV/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$BUILD_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="$BUILD_ENV/lib/pkgconfig:$BUILD_ENV/share/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

pkg-config --exists tcl libffi

if [[ ${1:-} == --check-deps ]]; then
  check_bin="${TMPDIR:-/tmp}/yosys-deps-check.$$"
  trap 'rm -f "$check_bin"' EXIT
  printf '%s\n' \
    '#include <ffi.h>' \
    '#include <readline/readline.h>' \
    '#include <tcl.h>' \
    '#include <zlib.h>' \
    'int main() { Tcl_FindExecutable(nullptr); return 0; }' \
    | g++ -std=c++20 -x c++ - $(pkg-config --cflags --libs tcl libffi) \
      -lreadline -lz -o "$check_bin"
  "$check_bin"
  echo "Yosys build dependencies are available."
  exit 0
fi

echo "started=$(date --iso-8601=seconds) host=$(hostname)"
echo "cpus=$SLURM_CPUS_PER_TASK prefix=$PREFIX"
echo "compiler=$(g++ --version | sed -n '1p')"
echo "build_env=$BUILD_ENV tcl=$(pkg-config --modversion tcl)"

make config-gcc
printf 'CXXFLAGS += -I%s/include\nLINKFLAGS += -L%s/lib -Wl,-rpath,%s/lib\n' \
  "$BUILD_ENV" "$BUILD_ENV" "$BUILD_ENV" >> Makefile.conf
make -j"$SLURM_CPUS_PER_TASK" PREFIX="$PREFIX"
make PREFIX="$PREFIX" install
"$PREFIX/bin/yosys" -V

mkdir -p "$SMOKE"
cd "$REPO"
"$PYTHON" benchmarks/pipeline/import_verilog.py \
  tests/fixtures/pipeline/tiny_mul.v \
  --top tiny_mul \
  --yosys "$PREFIX/bin/yosys" \
  --source-id repository-fixture \
  --source-revision b790dae43b74549c2c5aadb852f944f96aee3274 \
  --architecture array-ripple \
  --out "$SMOKE/tiny-mul.json"
"$PYTHON" benchmarks/pipeline/circuitsat_to_cnf.py \
  "$SMOKE/tiny-mul.json" \
  --out "$SMOKE/tiny-mul.cnf"
"$PYTHON" benchmarks/pipeline/validate.py \
  --circuitsat "$SMOKE/tiny-mul.json" \
  --cnf "$SMOKE/tiny-mul.cnf"
"$PYTHON" benchmarks/scope/audit.py audit \
  benchmarks/scope/benchmark-scope.yaml \
  benchmarks/scope/benchmark-scope.lock

echo "finished=$(date --iso-8601=seconds)"
