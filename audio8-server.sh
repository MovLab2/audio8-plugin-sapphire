#!/usr/bin/env bash
# Audio8 as a GPU server on its own, no Sapphire on this computer.
# Another computer's Sapphire uses it with Settings > Audio8 > Client only.
# Every start prints the same pairing string: paste it there once.
# Everything it keeps lands in ./user/ beside this file (its certificate,
# the keys of the computers let in, the voices they sent); the model (~2.5 GB)
# goes to the Hugging Face cache. Inside a Sapphire install
# (user/plugins/audio8) it uses that Sapphire's user/ instead: never files in
# the plugin's own folder, which its signature would refuse.
#
#   ./audio8-server.sh               run it (builds its conda env the first time, ~4 GB)
#   ./audio8-server.sh add <name>    let another computer in: prints its pairing string
#   ./audio8-server.sh remove <name> shut one out
#   ./audio8-server.sh list          who is let in
#
# PORT (5015) is the door other computers use; open it in your firewall.
set -euo pipefail
cd "$(dirname "$0")"
ENV=${AUDIO8_ENV:-audio8-server}     # AUDIO8_ENV: use an env you already have
PORT=${PORT:-5015}
ROOT="$PWD"
[ -f ../../../sapphire.py ] && ROOT="$(cd ../../.. && pwd)"
DOOR="$ROOT/user/worker_door/audio8"

command -v conda >/dev/null || { echo "conda not found: install Miniconda, then run this again"; exit 1; }
eval "$(conda shell.bash hook)"

if ! conda env list | grep -q "^$ENV "; then
    echo "Making the $ENV environment (once)..."
    conda create -y -n "$ENV" python=3.12 --override-channels -c conda-forge   # conda-forge: no Anaconda ToS prompt to block a script
fi
conda activate "$ENV"

# The packages are the ones plugin.json lists, read from it: one list, never two
# (plus cryptography for the door). What was installed is noted inside the
# env, so a remade env installs again.
mkdir -p "$ROOT/user"
STAMP="$CONDA_PREFIX/audio8-requirements.txt"
NEW="$ROOT/user/audio8-requirements.new"
python -c "import json; print('\n'.join(json.load(open('plugin.json', encoding='utf-8'))['environment']['pip'] + ['cryptography']))" > "$NEW"
if ! cmp -s "$NEW" "$STAMP"; then
    echo "Installing packages (the first time downloads ~4 GB)..."
    python -m pip install -r "$NEW"
    mv "$NEW" "$STAMP"
else
    rm "$NEW"
fi

case "${1:-run}" in
    add)    python worker_door.py add "$DOOR" "${2:?a name, like laptop}" "$PORT" ;;
    remove) python worker_door.py remove "$DOOR" "${2:?which name}" ;;
    list)   python worker_door.py list "$DOOR" ;;
    run)
        export SAPPHIRE_ROOT="$ROOT" SAPPHIRE_SERVICE_PORT=5014
        export SAPPHIRE_PLUGIN_SETTINGS="{\"share\": true, \"share_port\": $PORT}"
        exec python server.py ;;
    *) sed -n '2,14p' "$0"; exit 2 ;;
esac
