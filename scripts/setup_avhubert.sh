uv sync
git clone https://github.com/facebookresearch/av_hubert
cd av_hubert
git submodule init
git submodule update
cd fairseq
CXX=g++ CC=gcc uv pip install .
cd ../..
wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/base_vox_433h.pt -O assets/av_hubert.pt
CXX=g++ CC=gcc uv run scripts/test_av_hubert_installation.py 