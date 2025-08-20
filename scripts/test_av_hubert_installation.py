# /// script
# requires-python = "~=3.9"
# dependencies = [
#   "numpy<1.20",
#   "scipy==1.10.0",
#   "opencv-python",
#   "torch<2.6",
#   "omegaconf==2.1.1",
#   "hydra-core==1.1.1",
#   "python-speech-features==0.6",
#   "scikit-image",
#    "tqdm",
#    "sentencepiece==0.1.96"
# ]
# ///
from pathlib import Path
import sys

AVP_LIB_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(AVP_LIB_PATH))

print("Testing AV hubert setup")
try:
    import av_preprocessing as avp
    model = avp.av_hubert.instantiate_av_hubert_and_add_to_python_path()

    print("AV hubert setup successful")
except Exception as e:
    print("AV hubert setup failed: ", e)
    raise e
