#!/usr/bin/env bash
# Idempotent EDA + web-vendor setup for DLNN phase 2 (P0). Uses ONLY the sources approved by Blake:
#   1. apt PPA ppa:kicad/kicad-9.0-releases          (KiCad 9 + its symbol/footprint/3D libraries)
#   2. github.com/freerouting/freerouting built with its own Gradle wrapper (Maven Central / Gradle plugin portal / the
#      Temurin JDK 25 that the freerouting build itself requests via its foojay toolchain resolver)
#   3. npm: d3@7.9.1 three@0.187.0 eecircuit-engine@1.9.0; KiCanvas official build (kicanvas.org) + LICENSE from
#      github.com/theacodes/kicanvas
#   4. pip into hw/.venv-eda (python3.12, --system-site-packages): kicad-sch-api==0.5.6 kinet2pcb kiutils skidl
# Network goes through the pre-configured proxy; TLS verification is never disabled.
# Steps: ./setup_eda.sh [kicad|venv|freerouting|vendor|all]   (default all). Re-running skips finished steps.
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)
STEPS=${*:-all}
FR_DIR=${FR_DIR:-/home/user/freerouting/freerouting}
VENDOR=docs/artifacts/vendor
want() { [[ " $STEPS " == *" all "* || " $STEPS " == *" $1 "* ]]; }
say() { printf '== %s\n' "$*"; }

step_kicad() {
  say "KiCad 9 (PPA ppa:kicad/kicad-9.0-releases)"
  if command -v kicad-cli >/dev/null && [[ "$(kicad-cli version)" == 9.* ]] && [ -d /usr/share/kicad/symbols ] && [ -d /usr/share/kicad/3dmodels ]; then
    echo "already present: kicad-cli $(kicad-cli version)"
  else
    # add-apt-repository needs the system python (apt_pkg); the default python3 may be a different build
    if ! ls /etc/apt/sources.list.d 2>/dev/null | grep -q kicad; then
      add-apt-repository -y ppa:kicad/kicad-9.0-releases 2>/dev/null || /usr/bin/python3.12 /usr/bin/add-apt-repository -y ppa:kicad/kicad-9.0-releases
    fi
    apt-get update -qq
    # --no-install-recommends skips the libraries, which kicad-cli needs for symbols, footprints and 3D models
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends kicad kicad-symbols kicad-footprints kicad-packages3d
  fi
  mkdir -p ~/.config/kicad/9.0
  for t in sym-lib-table fp-lib-table; do   # global library tables (the headless first run only creates an empty fp table)
    grep -q 'KICAD9_.*_DIR' ~/.config/kicad/9.0/$t 2>/dev/null || cp /usr/share/kicad/template/$t ~/.config/kicad/9.0/$t
  done
  python3.12 -c "import pcbnew; print('pcbnew', pcbnew.Version())"
}

step_venv() {
  say "hw/.venv-eda (python3.12, system site packages)"
  [ -x hw/.venv-eda/bin/python ] || /usr/bin/python3.12 -m venv --system-site-packages hw/.venv-eda
  hw/.venv-eda/bin/pip install -q kicad-sch-api==0.5.6 kinet2pcb kiutils skidl
  hw/.venv-eda/bin/pip list 2>/dev/null | grep -iE '^(kicad-sch-api|kinet2pcb|kiutils|skidl) '
}

step_freerouting() {
  say "freerouting"
  if [ ! -d "$FR_DIR/.git" ]; then
    mkdir -p "$(dirname "$FR_DIR")"
    GIT_LFS_SKIP_SMUDGE=1 timeout 600 git clone --depth 1 https://github.com/freerouting/freerouting "$FR_DIR"
  fi
  if [ ! -f "$FR_DIR/build/libs/freerouting-current-executable.jar" ]; then
    # Maven Central sometimes answers 429 through the proxy: retry (Gradle caches what it already fetched)
    for i in 1 2 3 4 5 6; do
      (cd "$FR_DIR" && ./gradlew --no-daemon executableJar -x test) && break
      echo "gradle attempt $i failed, retrying in 20 s"; sleep 20
    done
  fi
  ls -la "$FR_DIR/build/libs/freerouting-current-executable.jar"
  echo "freerouting commit: $(git -C "$FR_DIR" rev-parse --short HEAD); JVM (toolchain JDK 25, Java 21 is too old for these classes):"
  ls -d ~/.gradle/jdks/*/bin/java 2>/dev/null
}

step_vendor() {
  say "web vendor -> $VENDOR"
  mkdir -p $VENDOR/{eecircuit-engine,d3,three/addons/{loaders,utils,controls},kicanvas}
  local tmp; tmp=$(mktemp -d)
  if [ ! -f $VENDOR/eecircuit-engine/eecircuit-engine.mjs ] || [ ! -f $VENDOR/d3/d3.min.js ] || [ ! -f $VENDOR/three/three.module.js ]; then
    (cd "$tmp" && for p in eecircuit-engine@1.9.0 d3@7.9.1 three@0.187.0; do
       npm pack "$p" >/dev/null && d=x_${p%@*} && mkdir -p "$d" && tar xzf "${p%@*}"-*.tgz -C "$d" || exit 1
       rm -f ./*.tgz; done)
    cp "$tmp/x_eecircuit-engine/package/dist/eecircuit-engine.mjs" "$tmp/x_eecircuit-engine/package/LICENSE" $VENDOR/eecircuit-engine/
    cp "$tmp/x_d3/package/dist/d3.min.js" "$tmp/x_d3/package/LICENSE" $VENDOR/d3/
    T=$tmp/x_three/package
    cp $T/build/three.module.js $T/build/three.core.js $T/LICENSE $VENDOR/three/
    cp $T/examples/jsm/loaders/GLTFLoader.js $VENDOR/three/addons/loaders/
    cp $T/examples/jsm/utils/BufferGeometryUtils.js $T/examples/jsm/utils/SkeletonUtils.js $VENDOR/three/addons/utils/
    cp $T/examples/jsm/controls/OrbitControls.js $VENDOR/three/addons/controls/
  else echo "npm packages already vendored"; fi
  if [ ! -f $VENDOR/kicanvas/kicanvas.js ]; then
    curl -fsS -o $VENDOR/kicanvas/kicanvas.js https://kicanvas.org/kicanvas/kicanvas.js
    curl -fsS -o $VENDOR/kicanvas/LICENSE https://raw.githubusercontent.com/theacodes/kicanvas/main/LICENSE.md
  else echo "kicanvas already vendored"; fi
  rm -rf "$tmp"
  python3 - <<'PYEOF'
import gzip, hashlib, json, pathlib
V = pathlib.Path("docs/artifacts/vendor")
spec = {
 "eecircuit-engine": ("1.9.0", "npm eecircuit-engine@1.9.0", "MIT", ["eecircuit-engine/eecircuit-engine.mjs"]),
 "d3": ("7.9.1", "npm d3@7.9.1", "ISC", ["d3/d3.min.js"]),
 "three": ("0.187.0", "npm three@0.187.0", "MIT", ["three/three.module.js", "three/three.core.js", "three/addons/loaders/GLTFLoader.js",
          "three/addons/utils/BufferGeometryUtils.js", "three/addons/utils/SkeletonUtils.js", "three/addons/controls/OrbitControls.js"]),
 "kicanvas": ("official build, no version string (package.json says 0.0.0)", "https://kicanvas.org/kicanvas/kicanvas.js", "MIT", ["kicanvas/kicanvas.js"]),
}
out = {}
for n, (ver, src, lic, files) in spec.items():
    out[n] = {"version": ver, "source": src, "license": lic, "files": {}}
    for f in files:
        b = (V / f).read_bytes()
        out[n]["files"][f] = {"bytes": len(b), "gzip_bytes": len(gzip.compress(b, 9)), "sha256": hashlib.sha256(b).hexdigest()}
(V / "VENDOR.json").write_text(json.dumps(out, indent=1) + "\n")
print("wrote", V / "VENDOR.json")
PYEOF
}

want kicad && step_kicad
want venv && step_venv
want freerouting && step_freerouting
want vendor && step_vendor
say "done"
