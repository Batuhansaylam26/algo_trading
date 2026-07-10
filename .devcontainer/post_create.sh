#!/bin/sh

warn() {
  printf '\n[postCreate warning] %s\n' "$*" >&2
}

run_optional() {
  label="$1"
  shift
  if "$@"; then
    printf '[postCreate] %s ok\n' "$label"
  else
    warn "$label failed; container will still open"
  fi
}

apt-get update
apt-get install -y postgresql-client curl gzip libgomp1 rsync
printf '[postCreate] apt packages ok\n'

if ! ldconfig -p | grep -q 'libgomp.so.1'; then
  warn "libgomp.so.1 is still missing after apt install; LightGBM/MLForecast will fail"
fi

if ! command -v duckdb >/dev/null 2>&1; then
  if curl -fsSL https://install.duckdb.org -o /tmp/install_duckdb.sh; then
    run_optional "duckdb install" sh /tmp/install_duckdb.sh
    if [ -x /root/.duckdb/cli/latest/duckdb ]; then
      run_optional "duckdb link" ln -sf /root/.duckdb/cli/latest/duckdb /usr/local/bin/duckdb
    fi
  else
    warn "duckdb install script download failed; install it manually if needed"
  fi
fi

run_optional "runtime bootstrap" python scripts/bootstrap_runtime.py

if [ ! -x .venv/bin/python ]; then
  run_optional "venv create" python -m venv .venv
fi

if [ -x .venv/bin/python ]; then
  . .venv/bin/activate
  run_optional "pip upgrade" pip install --no-cache-dir --upgrade pip
  run_optional "torch cpu preinstall" pip install --no-cache-dir --force-reinstall --index-url https://download.pytorch.org/whl/cpu torch
  if [ -f /opt/requirements-runtime.txt ]; then
    run_optional "runtime requirements" pip install --no-cache-dir -r /opt/requirements-runtime.txt
  else
    warn "/opt/requirements-runtime.txt not found; run python scripts/bootstrap_runtime.py manually"
  fi
  if command -v dbt >/dev/null 2>&1; then
    run_optional "dbt parse" dbt parse --profiles-dir /opt/dbt_project --project-dir /opt/dbt_project
  else
    warn "dbt command not found after install"
  fi
else
  warn ".venv/bin/python not found; venv setup failed"
fi

exit 0
