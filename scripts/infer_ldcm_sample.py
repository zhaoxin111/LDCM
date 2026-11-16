"""Run the official LDCM sample_1 with local checkpoints and isolated utils3d."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from infer_odin1_ldcm import runtime, provenance


def main():
    args=SimpleNamespace(ldcm_root=ROOT,checkpoint=ROOT/'weights/ldcm/ldcm.pt',
                         moge=ROOT/'weights/moge-2-vits-normal/model.pt')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/ldcm_sample_1')
    output=parser.parse_args().output
    if output.exists():
        raise FileExistsError(output)
    torch,Model=runtime(args.ldcm_root)
    import numpy as np
    from main import load_image,load_depth,save_preview
    torch.set_num_threads(4)
    sample=args.ldcm_root/'assets/sample_1'
    image=load_image(sample/'image.png')
    prior=load_depth(sample/'sparse_depth.npy')
    started=time.perf_counter()
    model=Model.from_pretrained(str(args.checkpoint),str(args.moge)).eval()
    load_seconds=time.perf_counter()-started
    started=time.perf_counter()
    with torch.inference_mode():
        result=model.infer(image,prior)
    seconds=time.perf_counter()-started
    pred=result['depth_pred'][0,0].float().numpy()
    mask=result['mask'][0,0].numpy()
    assert pred.shape==prior.shape[-2:] and np.isfinite(pred).all() and (pred>0).all() and mask.any()
    coarse=result['coarse_depth'].numpy()
    assert np.isfinite(coarse).all() and (coarse>0).any()
    output.mkdir(parents=True)
    np.save(output/'depth_pred.npy',pred)
    np.save(output/'mask.npy',mask)
    np.save(output/'coarse_depth.npy',coarse)
    save_preview(sample/'image.png',prior[0,0].numpy(),pred,sample/'gt_depth.npy',output/'preview.png')
    meta=dict(provenance=provenance(args),threads=4,device='cpu',model_load_seconds=load_seconds,
              inference_seconds=seconds,prediction_shape=list(pred.shape),valid_fraction=float(mask.mean()),
              poisson_solver='PyTorch CPU',validation_passed=True)
    (output/'manifest.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:v for k,v in meta.items() if k!='provenance'},indent=2),flush=True)


if __name__=='__main__':
    main()
