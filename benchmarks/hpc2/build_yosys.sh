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
PYTHON="$REPO/.venv/bin/python"
SMOKE=/hpc2ssd/JH_DATA/spooler/xpan432/boolean-inference/data/setup-smoke

echo "started=$(date --iso-8601=seconds) host=$(hostname)"
echo "cpus=$SLURM_CPUS_PER_TASK prefix=$PREFIX"
echo "compiler=$(g++ --version | sed -n '1p')"

make config-gcc
printf 'ENABLE_READLINE := 0\n' >> Makefile.conf
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
