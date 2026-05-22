set -e

# GREEN='\033[0;32m'
# BLUE='\033[0;34m'
# NC='\033[0m' # No Color

# ---- 小工具 ----
log() { printf "\n\033[1;32m[+] %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m[!] %s\033[0m\n" "$*"; }
die() { printf "\033[1;31m[x] %s\033[0m\n" "$*"; exit 1; }
trap 'die "脚本在第 $LINENO 行出错（exit=$?）。"' ERR

eval "$(conda shell.bash hook)"
conda create -n agentrl_web python==3.10 pip -c conda-forge
conda activate agentrl_web
python3 -m pip install uv

log "安装 requirements_webshop.txt"
pip install -r requirements_webshop.txt

log "安装 faiss-cpu"
conda install -c pytorch faiss-cpu -y
# sudo apt install -y default-jdk
conda install -c conda-forge openjdk=21 maven -y
conda install mkl -y

# webshop installation, model loading
log "安装 spacy, if failed, try to install spacy from source version"
# pip install -U "numpy==2.2.6" "scipy==1.15.3"
# python -m spacy download en_core_web_sm --direct
# python -m spacy download en_core_web_lg --direct
# python3 -m uv pip install -U "numpy==2.2.6"

log "gdown files from google drive"
conda install conda-forge::gdown -y
mkdir -p dataset;
cd dataset;
# [ -f items_shuffle.json ] || gdown https://drive.google.com/uc?id=1Yk4hsehueKuWwDCCsUOyyEpqUkzrmUu1 -O items_shuffle.json
# # items_ins_v2
# [ -f items_ins_v2.json ] || gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi -O items_ins_v2.json
# items_shuffle_1000 - product scraped info
# [ -f items_shuffle_1000.json ] || gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib -O items_shuffle_1000.json 
# # items_ins_v2_1000 - product attributes
# [ -f items_ins_v2_1000.json ] || gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu -O items_ins_v2_1000.json
# # items_human_ins
# [ -f items_human_ins.json ] || gdown https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O -O items_human_ins.json
cd ..

log "Build search engine index"
cd search_engine
mkdir -p resources resources_100 resources_1k resources_100k
python convert_product_file_format.py # convert items.json => required doc format
mkdir -p indexes
./run_indexing.sh
cd ..

log "安装 verl package"
cd ../../../../../
# python3 -m uv pip install -e ".[sglang]"
# python3 -m uv pip install -e ".[vllm]"
python3 -m uv pip install --upgrade vllm==0.8.5
pip install --no-deps -e .
# python3 -m uv pip install flash-attn==2.8.3 --no-build-isolation --no-deps
python3 -m uv pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -r requirements.txt
python3 -m uv pip install ray==2.45.0
python3 -m uv pip install click==8.2.1


echo -e "${GREEN}Installation completed successfully!${NC}"
