uv sync
git clone https://github.com/facebookresearch/av_hubert
cd av_hubert
git submodule init
git submodule update
cd fairseq
echo "Installing fairseq"
CXX=g++ CC=gcc uv pip install .
cd ../..
wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/base_vox_433h.pt -O assets/av_hubert.pt
echo "Testing AV-Hubert installation"
CXX=g++ CC=gcc uv run --python 3.9 scripts/test_av_hubert_installation.py 