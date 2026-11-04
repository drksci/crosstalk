#!/usr/bin/env bash
# Idempotent environment setup for the DLNN PoC. Safe to re-run: every step first checks whether it is already done.
# Network goes through the pre-configured proxy; TLS verification is never disabled.
# Each component is recorded as INSTALLED / already present / FAILED in the ENV block of docs/build-log.md.
#
#   ./setup.sh                 core + rf + torch + llm (llama-cpp-python built from sdist, ~4 min the first time) + Qwen weights + psycopg
#   WITH_LLM=0 ./setup.sh      skip llama-cpp-python build and the Qwen weights download (~1.1 GB)
#   WITH_TORCH=0 ./setup.sh    skip torch (only needed for training: python -m dlnn.model.train)
#   WITH_PG=1 ./setup.sh       also apt-install postgresql (a local server for --db postgres)
set -u
cd "$(dirname "$0")"
LOG=docs/build-log.md
mkdir -p docs models/llm
ENVLOG=$(mktemp)
rec() { echo "- $*" >> "$ENVLOG"; }

WITH_LLM=${WITH_LLM:-1}; WITH_TORCH=${WITH_TORCH:-1}; WITH_PG=${WITH_PG:-0}

rec "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
rec "python: $(python3 --version 2>&1)"

# --- venv + core pip packages (pyproject extras: rf, dev, pg) ---------------------------------------------
[ -d .venv ] || python3 -m venv .venv
PIP=.venv/bin/pip
PY=.venv/bin/python
if $PIP install -q --upgrade pip && $PIP install -q -e ".[rf,dev,pg]" ; then
  rec "pip core + extras rf,dev,pg (numpy scipy matplotlib jsonschema nbformat nbconvert papermill ipykernel jupyter scikit-rf pandas pytest psycopg[binary]; editable dlnn): INSTALLED"
  PIPOK=1
else
  rec "pip core + extras rf,dev,pg: FAILED (see terminal output)"; PIPOK=0
fi

# --- torch (CPU wheel from the pytorch index; extra 'train' names it, the index URL cannot live in pyproject) ----
if [ "$WITH_TORCH" = 1 ]; then
  if $PY -c "import torch" >/dev/null 2>&1; then
    rec "torch: already present ($($PY -c 'import torch;print(torch.__version__)'))"
  elif $PIP install -q torch --index-url https://download.pytorch.org/whl/cpu && $PY -c "import torch" >/dev/null 2>&1; then
    rec "torch: INSTALLED ($($PY -c 'import torch;print(torch.__version__)'))"
  else
    rec "torch: FAILED (training unavailable; shipped models in models/ still run)"
  fi
else
  rec "torch: skipped (WITH_TORCH=0)"
fi

# --- llama-cpp-python 0.3.19 (Qwen backend) -----------------------------------------------------------------
# Built from the pinned SDIST, never from the index wheel: the abetlen CPU wheel (cp313 linux_x86_64) is linked
# against musl (libc.musl-x86_64.so.1) and fails to import on a glibc box (see docs/llm_commentary.md, build-log-M3.md).
#   .venv/bin/pip install --no-binary llama-cpp-python --no-cache-dir llama-cpp-python==0.3.19
# Needs cmake and a C/C++ compiler (gcc 13 was used). Build time on this 4-core box with 2 jobs: about 4 minutes.
# -DGGML_NATIVE=ON ties the build to this CPU (-march=native); do not copy the venv to another machine.
# diskcache (a dependency of llama-cpp-python) comes in with it.
if [ "$WITH_LLM" = 1 ]; then
  if [ "$($PY -c 'import llama_cpp;print(llama_cpp.__version__)' 2>/dev/null)" = "0.3.19" ]; then
    rec "llama-cpp-python: already present (0.3.19)"
  elif ! command -v cmake >/dev/null 2>&1 || ! { command -v c++ >/dev/null 2>&1 || command -v g++ >/dev/null 2>&1 || command -v clang++ >/dev/null 2>&1; }; then
    rec "llama-cpp-python: FAILED (needs cmake and a C/C++ compiler to build from source; --llm qwen unavailable, --llm none works)"
  else
    T0=$(date +%s)
    CMAKE_BUILD_PARALLEL_LEVEL=2 CMAKE_ARGS="-DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF" \
      $PIP install -q --no-binary llama-cpp-python --no-cache-dir "llama-cpp-python==0.3.19" >/tmp/llama_build.log 2>&1
    if $PY -c "import llama_cpp" >/dev/null 2>&1; then
      rec "llama-cpp-python 0.3.19: INSTALLED (source build, $(( $(date +%s) - T0 )) s)"
    else
      rec "llama-cpp-python: FAILED ($(tail -n 1 /tmp/llama_build.log 2>/dev/null)); --llm qwen unavailable, --llm none works"
    fi
  fi
  Q=models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf
  if [ -s "$Q" ]; then
    rec "Qwen2.5-1.5B-Instruct Q4_K_M weights: already present ($Q)"
  elif curl -fsSL -o "$Q.part" "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
       && mv "$Q.part" "$Q"; then
    rec "Qwen2.5-1.5B-Instruct Q4_K_M weights: INSTALLED ($Q)"
  else
    rm -f "$Q.part"; rec "Qwen2.5-1.5B-Instruct Q4_K_M weights: FAILED (download)"
  fi
  # BitNet b1.58-2B-4T via bitnet.cpp is an OPTIONAL second backend and is NOT installed by this script. It built here but its
  # output was degenerate (0/5 injected cards attributed; docs/llm_eval.md), so it is not recommended. Commands that were run:
  #   git clone --recursive --depth 1 https://github.com/microsoft/BitNet.git models/llm/bitnet.cpp
  #   curl -L -o models/llm/bitnet/ggml-model-i2_s.gguf \
  #        https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf
  #   cd models/llm/bitnet.cpp && cmake -B build -DBITNET_X86_TL2=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  #        -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF
  #   cmake --build build --target llama-cli llama-completion -j2      # needs clang >= 18; about 20 min under load
  rec "BitNet b1.58-2B-4T (optional second backend): not installed by this script (commands in the setup.sh comments and docs/llm_eval.md)"
else
  rec "llama-cpp-python / Qwen weights: skipped (WITH_LLM=0)"
fi

# --- jupyter kernel ------------------------------------------------------------------------------------------
if [ "$PIPOK" = 1 ] && $PY -m ipykernel install --user --name dlnn --display-name "dlnn (.venv)" >/dev/null 2>&1; then
  rec "jupyter kernel 'dlnn': REGISTERED"
else
  rec "jupyter kernel 'dlnn': FAILED"
fi

# --- apt: ngspice, iverilog (and optionally postgresql) --------------------------------------------------------
PKGS="ngspice iverilog"; [ "$WITH_PG" = 1 ] && PKGS="$PKGS postgresql"
for pkg in $PKGS; do
  bin=$pkg; [ "$pkg" = postgresql ] && bin=psql
  if command -v "$bin" >/dev/null 2>&1; then
    rec "apt $pkg: already present ($(command -v $bin))"
  elif (apt-get install -y "$pkg" >/tmp/apt_$pkg.log 2>&1) && command -v "$bin" >/dev/null 2>&1; then
    rec "apt $pkg: INSTALLED ($(command -v $bin))"
  else
    rec "apt $pkg: FAILED ($(tail -n 1 /tmp/apt_$pkg.log 2>/dev/null))"
  fi
done
[ "$WITH_PG" = 1 ] || rec "postgresql server: not requested (WITH_PG=1 to install; --db sqlite is the default)"

# --- record into build-log.md under "Environment" (replace previous block) -----------------------------------------
touch "$LOG"
$PY - "$LOG" "$ENVLOG" <<'PY'
import sys, re
log, env = sys.argv[1], sys.argv[2]
body = open(env).read().rstrip("\n")
txt = open(log).read()
block = "<!-- ENV-BEGIN -->\n## Environment\n\n" + body + "\n<!-- ENV-END -->\n"
if "<!-- ENV-BEGIN -->" in txt:
    txt = re.sub(r"<!-- ENV-BEGIN -->.*?<!-- ENV-END -->\n", lambda m: block, txt, flags=re.S)
else:
    txt = (txt.rstrip("\n") + "\n\n" if txt.strip() else "# Build log\n\n") + block
open(log, "w").write(txt)
PY
rm -f "$ENVLOG"
echo "setup done; see $LOG"
