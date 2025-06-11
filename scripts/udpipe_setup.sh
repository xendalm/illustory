#!/bin/bash

git submodule update --init --recursive

python -m venv .venv

source .venv/bin/activate

pip install "tensorflow==2.12.1" "ufal.udpipe>=1.3,<2" ufal.chu_liu_edmonds transformers

if [ ! -d "ru_all-ud-2.15-241121.model" ]; then
    echo "ru_all-ud-2.15-241121.model NOT DOUND"
    exit 1
fi

python3 udpipe2_server.py 8001 --logfile udpipe2_server.log --threads=4 ru russian-syntagrus-ud-2.15-241121:ru_syntagrus-ud-2.15-241121:ru:rus ru_all-ud-2.15-241121.model ru_syntagrus https://ufal.mff.cuni.cz/udpipe/2/models#universal_dependencies_215_models
