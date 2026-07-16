#!/bin/bash
#SBATCH -J bi-scope-v3
#SBATCH -p i64m512u
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -D /hpc2hdd/home/xpan432/Codes/boolean-inference
#SBATCH -o /hpc2ssd/JH_DATA/spooler/xpan432/boolean-inference/logs/scope-v3-%j.out
#SBATCH -e /hpc2ssd/JH_DATA/spooler/xpan432/boolean-inference/logs/scope-v3-%j.err

set -euo pipefail

REPO=/hpc2hdd/home/xpan432/Codes/boolean-inference
PYTHON="$REPO/.venv/bin/python"
YOSYS=/hpc2ssd/JH_DATA/spooler/xpan432/boolean-inference/build/yosys-0.66/bin/yosys
BUILD_ENV=/hpc2hdd/home/xpan432/envs/yosys-0.66-build
MULTGEN=/hpc2hdd/home/xpan432/Codes/multgen/multgen
MULTGEN_REV=215fe0a77b2f3e61f6757a39323afa13bbe8e13f
DATA_ROOT=/hpc2ssd/JH_DATA/spooler/xpan432/boolean-inference/data
FINAL="$DATA_ROOT/scope-v3-candidate"
STAGE="$DATA_ROOT/scope-v3-candidate.partial.$SLURM_JOB_ID"
ARCHIVE=/hpc2hdd/JH_DATA/jhai_data/xpan432/boolean-inference/benchmarks/scope-v3-candidate.tar.zst
ARCHIVE_STAGE="$ARCHIVE.partial.$SLURM_JOB_ID"

test -x "$PYTHON"
test -x "$YOSYS"
test -x "$MULTGEN"
test "$(git -C "$(dirname "$MULTGEN")" rev-parse HEAD)" = "$MULTGEN_REV"
test ! -e "$FINAL"
test ! -e "$STAGE"
test ! -e "$ARCHIVE"
test ! -e "$ARCHIVE_STAGE"

export PATH="$BUILD_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$BUILD_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

report_failure() {
  echo "generation failed" >&2
  test ! -e "$STAGE" || echo "partial data retained at $STAGE" >&2
  test ! -e "$FINAL" || echo "validated data retained at $FINAL" >&2
  test ! -e "$ARCHIVE_STAGE" || echo "partial archive retained at $ARCHIVE_STAGE" >&2
}
trap report_failure ERR

mkdir -p "$STAGE/private" "$STAGE/raw" "$STAGE/verilog"

echo "started=$(date --iso-8601=seconds) host=$(hostname)"
echo "repo=$(git -C "$REPO" rev-parse HEAD)"
echo "yosys=$($YOSYS -V)"
echo "multgen=$MULTGEN_REV"

"$PYTHON" benchmarks/pipeline/generate_targets.py \
  --width 24 --width 32 --count 2 --seed-base 20260709 \
  --out "$STAGE/targets.jsonl" \
  --oracle-out "$STAGE/private/factor-oracle.jsonl"

for bits in 24 32; do
  "$PYTHON" benchmarks/pipeline/generate_targets.py \
    --width "$bits" --count 2 --seed-base 20260709 \
    --out "$STAGE/targets-$bits.jsonl"

  "$PYTHON" benchmarks/pipeline/generate_structural_multiplier.py \
    --bits "$bits" --architecture array-ripple \
    --out "$STAGE/raw/array-ripple-$bits.json"
  "$PYTHON" benchmarks/pipeline/generate_structural_multiplier.py \
    --bits "$bits" --architecture karatsuba --base-case 4 \
    --out "$STAGE/raw/karatsuba-$bits.json"
done

declare -A TREES=(
  [wallace-ripple]=WT
  [dadda-ripple]=DT
  [booth-r4-wallace]=WT
  [booth-r4-dadda]=DT
)
declare -A PARTIAL_PRODUCTS=(
  [wallace-ripple]=USP
  [dadda-ripple]=USP
  [booth-r4-wallace]=UB4
  [booth-r4-dadda]=UB4
)

for bits in 24 32; do
  for architecture in wallace-ripple dadda-ripple booth-r4-wallace booth-r4-dadda; do
    tree=${TREES[$architecture]}
    pp=${PARTIAL_PRODUCTS[$architecture]}
    top="${tree}_${pp}_RP_${bits}x${bits}_noX"
    prefix="$STAGE/verilog/$architecture-$bits-"
    source="$prefix${top}_multgen.sv"

    "$MULTGEN" -def -type StandAlone -tree "$tree" -pp "$pp" -adder RP \
      -in1size "$bits" -in2size "$bits" -outsize "$((bits * 2))" \
      -allowXes false -filenameprefix "$prefix"
    test -f "$source"

    "$PYTHON" benchmarks/pipeline/import_verilog.py "$source" \
      --top "$top" \
      --yosys "$YOSYS" \
      --source-id multgen \
      --source-revision "$MULTGEN_REV" \
      --architecture "$architecture" \
      --out "$STAGE/raw/$architecture-$bits.json"
  done
done

for bits in 24 32; do
  "$PYTHON" benchmarks/pipeline/generate_multiplier_instances.py \
    --targets "$STAGE/targets-$bits.jsonl" \
    --netlist "array-ripple=$STAGE/raw/array-ripple-{bits}.json" \
    --netlist "wallace-ripple=$STAGE/raw/wallace-ripple-{bits}.json" \
    --netlist "dadda-ripple=$STAGE/raw/dadda-ripple-{bits}.json" \
    --netlist "booth-r4-wallace=$STAGE/raw/booth-r4-wallace-{bits}.json" \
    --netlist "booth-r4-dadda=$STAGE/raw/booth-r4-dadda-{bits}.json" \
    --netlist "karatsuba=$STAGE/raw/karatsuba-{bits}.json" \
    --product-port wallace-ripple=result \
    --product-port dadda-ripple=result \
    --product-port booth-r4-wallace=result \
    --product-port booth-r4-dadda=result \
    --default-product-port product \
    --require-all \
    --out-dir "$STAGE/instances-$bits"
done

"$PYTHON" benchmarks/pipeline/collect_manifest.py \
  "$STAGE/instances-24" "$STAGE/instances-32" \
  --base "$STAGE" \
  --out "$STAGE/manifest.jsonl"

test "$(wc -l < "$STAGE/manifest.jsonl")" -eq 24
test "$(sha256sum "$STAGE/targets.jsonl" | cut -d ' ' -f 1)" = \
  4229e8295400ea075b989d0b4ab9d273ded0b32505bfe7002bbbcd6bbf9b3d69

"$PYTHON" benchmarks/pipeline/validate_multiplier_witnesses.py \
  --manifest "$STAGE/manifest.jsonl" \
  --oracle "$STAGE/private/factor-oracle.jsonl" \
  --raw-dir "$STAGE/raw"

"$PYTHON" benchmarks/scope/audit.py audit \
  benchmarks/scope/benchmark-scope.yaml \
  benchmarks/scope/benchmark-scope.lock

tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
  -I "$BUILD_ENV/bin/zstd -T1 -3" \
  --transform="s|^$(basename "$STAGE")|scope-v3-candidate|" \
  -cf "$ARCHIVE_STAGE" -C "$DATA_ROOT" "$(basename "$STAGE")"

mv "$STAGE" "$FINAL"
mv "$ARCHIVE_STAGE" "$ARCHIVE"
trap - ERR

echo "manifest_sha256=$(sha256sum "$FINAL/manifest.jsonl" | cut -d ' ' -f 1)"
echo "archive_sha256=$(sha256sum "$ARCHIVE" | cut -d ' ' -f 1)"
du -sh "$FINAL" "$ARCHIVE"
echo "finished=$(date --iso-8601=seconds)"
