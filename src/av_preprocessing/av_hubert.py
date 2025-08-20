import sys
import importlib.util
from pathlib import Path
from . import paths

def instantiate_av_hubert_and_add_to_python_path():
    try:
        assert paths.AV_HUBERT_LIB_PATH.exists(), "paths.AV_HUBERT_LIB_PATH does not exist"
        sys.path.append(str(paths.AV_HUBERT_LIB_PATH/"fairseq"))
        print(f"{paths.AV_HUBERT_LIB_PATH/'fairseq'} added to sys.path")
        DBG = True if len(sys.argv) == 1 else False
        # Fairseq has a debug mode depending on whther you pass sys.argv or not
        if DBG:  # sys.argv == 1
            sys.path.append(str(paths.AV_HUBERT_LIB_PATH / "avhubert"))
            print(f"{paths.AV_HUBERT_LIB_PATH} added to sys.path")
            import utils as avhubert_utils
            import hubert_pretraining, hubert, hubert_asr
        else:  # sys.argv > 1
            # Facebook's code is also overwritting the module avhubert.utils so we have to bypass it
            sys.path.append(str(paths.AV_HUBERT_LIB_PATH))
            print(f"{paths.AV_HUBERT_LIB_PATH} added to sys.path")

            spec = importlib.util.spec_from_file_location(
                "avhubert_utils", paths.AV_HUBERT_LIB_PATH / "avhubert" / "utils.py"
            )
            avhubert_utils = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(avhubert_utils)
            import avhubert
        from preparation.align_mouth import crop_patch

    except AssertionError:
        print("av_hubert not found")
    checkpoint_path = paths.ASSETS_PATH / "av_hubert.pt"
    # This shit happens due to facebook's implementation of the library
    #  https://github.com/facebookresearch/av_hubert/issues/36#issuecomment-1082194157
    # Do not delete the fairseq import
    import fairseq
    print(fairseq.__path__)
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([str(checkpoint_path)])
    model = models[0]
    return model, cfg, task