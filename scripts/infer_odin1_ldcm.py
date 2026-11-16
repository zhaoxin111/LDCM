"""Single-frame LDCM inference and matched DMDC comparison on an Odin1 bag."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DMDC_ROOT = ROOT.parent / "dToF-DMDC"
sys.path.insert(0, str(ROOT))
from odin1_common import (
    IMAGE_TOPIC, CLOUD_TOPIC, AnyReader, Stores, get_typestore, stamp_ns,
    read_samples, rectify_maps, project_cloud, write_ply, depth_binned_consistency,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def revision(root):
    return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()


def runtime(ldcm_root):
    sys.path.insert(0, str(ROOT / 'weights/ldcm_runtime'))
    sys.path.insert(0, str(ldcm_root.resolve()))
    import torch
    from ldcm import LDCMModel
    return torch, LDCMModel


def provenance(args):
    return dict(python=sys.version, source_revision=revision(ROOT), ldcm_revision=revision(args.ldcm_root),
                script_sha256=sha256(__file__),
                odin1_helpers_sha256=sha256(ROOT / 'scripts/odin1_common.py'),
                ldcm_source_sha256=sha256(args.ldcm_root / 'ldcm/ldcm.py'),
                ldcm_diff=subprocess.check_output(['git', '-C', str(args.ldcm_root), 'diff'], text=True),
                weights={name: dict(path=str(path.resolve()), sha256=sha256(path))
                         for name, path in [('ldcm', args.checkpoint), ('moge', args.moge)]},
                weight_sources=json.loads((ROOT / 'weights/ldcm_sources.json').read_text())
                    if (ROOT / 'weights/ldcm_sources.json').exists() else None,
                opencv_version=cv2.__version__,
                dependencies={name: importlib.metadata.version(name) for name in
                              ['torch', 'torchvision', 'numpy', 'scipy', 'matplotlib',
                               'rosbags', 'PyYAML', 'huggingface_hub', 'utils3d']},
                utils3d_origin=importlib.metadata.distribution('utils3d').read_text('direct_url.json'))


def exact_samples(bag, records, max_delta):
    wanted = {IMAGE_TOPIC: {r['image_stamp_ns'] for r in records},
              CLOUD_TOPIC: {r['cloud_stamp_ns'] for r in records}}
    found = {}
    with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        connections = [c for c in reader.connections if c.topic in wanted]
        for c, _, data in reader.messages(connections=connections):
            msg = reader.deserialize(data, c.msgtype)
            key = stamp_ns(msg)
            if key in wanted[c.topic]:
                found[c.topic, key] = msg
    result = []
    for rec in records:
        it, ct = rec['image_stamp_ns'], rec['cloud_stamp_ns']
        if abs(it - ct) > max_delta * 1e9:
            raise ValueError('Reference timestamp pair exceeds maximum delta')
        result.append((found[IMAGE_TOPIC, it], found[CLOUD_TOPIC, ct], rec['fraction']))
    return result


def validate_ply(path, depth, rgb, k, valid):
    with path.open('rb') as stream:
        header = b''
        while not header.endswith(b'end_header\n'):
            line = stream.readline()
            if not line:
                raise ValueError('Incomplete PLY header')
            header += line
        data = np.fromfile(stream, dtype=[('xyz', '<f4', (3,)), ('rgb', 'u1', (3,))])
    v, u = np.nonzero(valid)
    expected = np.stack([(u-k[0, 2])*depth[v,u]/k[0,0],
                         (v-k[1, 2])*depth[v,u]/k[1,1], depth[v,u]], axis=1)
    assert f'element vertex {len(v)}\n'.encode() in header
    assert len(data) == len(v) > 0
    np.testing.assert_allclose(data['xyz'], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(data['rgb'], rgb[v,u])
    return dict(points=len(v), max_backprojection_error_m=float(np.max(np.abs(data['xyz']-expected))))


def resolve_reference(reference_dir, bag, frames):
    """Auto-compare only when the existing reference matches this bag/count."""
    automatic = reference_dir is None
    if automatic:
        reference_dir = DMDC_ROOT / 'outputs/03'
    elif reference_dir == Path('.'):
        return None, None
    manifest = reference_dir / 'manifest.json'
    if not manifest.is_file():
        if automatic:
            return None, None
        raise ValueError('Reference manifest missing; use --reference "" for independent frame selection')
    reference = json.loads(manifest.read_text())
    if Path(reference['arguments']['bag']).resolve() != bag.resolve() or frames > len(reference['frames']):
        if automatic:
            return None, None
        raise ValueError('Reference bag differs or requested frame count exceeds reference')
    return reference_dir, reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bag', type=Path, default=Path('/media/xzhao/nas163/data_xzhao/odin1/odin_bags/03'))
    p.add_argument('--calib', type=Path, default=Path('/media/xzhao/nas163/data_xzhao/odin1/calib.yaml'))
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/ldcm_03')
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'weights/ldcm/ldcm.pt')
    p.add_argument('--moge', type=Path, default=ROOT / 'weights/moge-2-vits-normal/model.pt')
    p.add_argument('--ldcm-root', type=Path, default=ROOT)
    p.add_argument('--reference', type=Path, default=None,
                   help='DMDC results and exact timestamps; default: auto-use 03 reference only for a matching bag/count. Empty path disables comparison')
    p.add_argument('--frames', type=int, default=3)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--width', type=int, default=800)
    p.add_argument('--height', type=int, default=648)
    p.add_argument('--fov-x', type=float, default=90.)
    p.add_argument('--confidence', type=int, default=35)
    p.add_argument('--min-range', type=float, default=.1)
    p.add_argument('--max-time-delta', type=float, default=.05)
    a = p.parse_args()
    if min(a.frames,a.threads,a.width,a.height) < 1 or not 0 < a.fov_x < 180 or not np.isfinite([a.min_range,a.max_time_delta]).all() or min(a.min_range,a.max_time_delta) < 0:
        p.error('Invalid frame count, threads, dimensions, FoV, range or time delta')
    if a.output.exists():
        p.error('Output exists; choose a new directory to preserve existing results')
    for path in [a.checkpoint,a.moge,a.calib,a.ldcm_root / 'ldcm/ldcm.py']:
        if not path.is_file():
            p.error(f'Missing local input: {path}')
    try:
        a.reference, reference = resolve_reference(a.reference, a.bag, a.frames)
    except ValueError as error:
        p.error(str(error))
    calib = yaml.safe_load(a.calib.read_text())
    cam = calib['cam_0']
    if cam['cam_model'] != 'FishPoly' or cam.get('p1',0) or cam.get('p2',0):
        raise ValueError('Expected FishPoly with zero tangential distortion')
    transform = np.asarray(calib['Tcl_0']).reshape(4,4)
    k,mx,my,image_valid = rectify_maps(cam,a.width,a.height,a.fov_x)
    if reference:
        np.testing.assert_array_equal(k,reference['intrinsics_px'])
        np.testing.assert_array_equal(transform,reference['T_camera_lidar'])
    print('Reading exact reference timestamp pairs...' if reference else 'Selecting frames...',flush=True)
    samples = exact_samples(a.bag,reference['frames'][:a.frames],a.max_time_delta) if reference else read_samples(a.bag,a.frames,a.max_time_delta)
    torch,LDCMModel = runtime(a.ldcm_root)
    torch.set_num_threads(a.threads)
    a.output.mkdir(parents=True)
    meta = dict(arguments={key:str(value) if isinstance(value,Path) else value for key,value in vars(a).items()},
                intrinsics_px=k.tolist(),T_camera_lidar=transform.tolist(),provenance=provenance(a),frames=[],
                coordinate_system='camera optical: x right, y down, z forward; meters',
                limitations='Single-frame offline inference; no motion compensation or fusion. Sparse input consistency is not ground-truth accuracy.')
    def save():
        (a.output / 'manifest.json').write_text(json.dumps(meta,indent=2,allow_nan=False))
    save()
    started=time.perf_counter()
    model=LDCMModel.from_pretrained(str(a.checkpoint.resolve()),str(a.moge.resolve())).to(a.device).eval()
    meta['model_load_seconds']=time.perf_counter()-started
    prepared=[]
    validations=[]
    for index,(imsg,cmsg,fraction) in enumerate(samples):
        folder=a.output / f'frame_{index:02d}'
        folder.mkdir()
        original=cv2.imdecode(np.asarray(imsg.data,dtype=np.uint8),cv2.IMREAD_COLOR)
        if original is None or original.shape[:2] != (cam['image_height'],cam['image_width']):
            raise ValueError('RGB dimensions differ from calibration')
        image=cv2.remap(original,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        sparse,stats=project_cloud(cmsg,transform,k,a.width,a.height,a.confidence,a.min_range)
        sparse[~image_valid]=np.nan
        if not np.isfinite(sparse).any():
            raise ValueError('No projected sparse depth')
        dmdc=None
        if reference:
            ref_folder=a.reference / f"frame_{reference['frames'][index]['frame']:02d}"
            np.testing.assert_array_equal(sparse,np.load(ref_folder / 'sparse_depth.npy'))
            np.testing.assert_array_equal(image,cv2.imread(str(ref_folder / 'rgb.png')))
            dmdc=np.load(ref_folder / 'depth.npy')
            assert dmdc.shape == sparse.shape
        rgb=image[...,::-1].copy()
        inp=torch.from_numpy(rgb).permute(2,0,1)[None].float().to(a.device)/255
        prior=torch.from_numpy(np.nan_to_num(sparse,nan=0.))[None,None].to(a.device)
        started=time.perf_counter()
        with torch.inference_mode():
            output=model.infer(inp,prior)
        if a.device.startswith('cuda'):
            torch.cuda.synchronize()
        seconds=time.perf_counter()-started
        depth=output['depth_pred'][0,0].float().cpu().numpy()
        model_mask=output['mask'][0,0].cpu().numpy().astype(bool)
        assert depth.shape == model_mask.shape == sparse.shape
        valid=image_valid & model_mask & np.isfinite(depth) & (depth>0)
        if not valid.any():
            raise ValueError('No valid prediction')
        depth=np.where(valid,depth,np.nan).astype(np.float32)
        for name,array in [('rgb',image),('mask',valid.astype(np.uint8)*255),('model_mask',model_mask.astype(np.uint8)*255),('rectification_mask',image_valid.astype(np.uint8)*255)]:
            assert cv2.imwrite(str(folder / f'{name}.png'),array)
        np.save(folder / 'depth.npy',depth)
        np.save(folder / 'sparse_depth.npy',sparse)
        np.testing.assert_array_equal(np.load(folder/'depth.npy'),depth)
        np.testing.assert_array_equal(cv2.imread(str(folder/'mask.png'),0)>0,valid)
        count=write_ply(folder / 'pointcloud.ply',depth,rgb,k,valid)
        write_ply(folder / 'sparse_pointcloud.ply',sparse,rgb,k,np.isfinite(sparse))
        rec=dict(frame=index,fraction=fraction,image_stamp_ns=stamp_ns(imsg),cloud_stamp_ns=stamp_ns(cmsg),
                 time_delta_ms=(stamp_ns(cmsg)-stamp_ns(imsg))/1e6,cloud_frame=cmsg.header.frame_id,**stats,
                 sparse_pixels=int(np.isfinite(sparse).sum()),inference_seconds=seconds,dense_points=count,
                 valid_fraction=float(valid.mean()),depth_percentiles_m=np.nanpercentile(depth,[0,1,50,99,100]).tolist(),
                 depth_binned_consistency=depth_binned_consistency(depth,sparse,valid))
        if dmdc is not None:
            common=valid & np.isfinite(dmdc) & (dmdc>0)
            assert common.any()
            rec['comparison']=dict(common_pixels=int(common.sum()),common_fraction=float(common.mean()),
                dmdc_inference_seconds=reference['frames'][index]['inference_seconds'],
                dmdc_valid_fraction=float((np.isfinite(dmdc)&(dmdc>0)).mean()),
                prediction_difference_mae_m=float(np.abs(depth[common]-dmdc[common]).mean()),
                ldcm_sparse_consistency_common=depth_binned_consistency(depth,sparse,common),
                dmdc_sparse_consistency_common=depth_binned_consistency(dmdc,sparse,common))
        validations.append(dict(frame=index,input_equal_to_dmdc=True if reference else None,
                                prediction_shape=list(depth.shape),
                                dense=validate_ply(folder/'pointcloud.ply',depth,rgb,k,valid),
                                sparse=validate_ply(folder/'sparse_pointcloud.ply',sparse,rgb,k,np.isfinite(sparse))))
        meta['frames'].append(rec)
        save()
        prepared.append((folder,rgb,sparse,depth,dmdc))
        print(f'Frame {index}: {seconds:.2f}s, coverage {valid.mean():.2%}, {count} points',flush=True)
        del output
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'ldcm-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values=np.concatenate([d[np.isfinite(d)&(d>0)] for _,_,_,pred,ref in prepared for d in [pred,ref] if d is not None])
    lo,hi=np.percentile(values,[1,99]); hi=max(hi,lo+1e-6)
    meta['visualization_range_m']=[float(lo),float(hi)]
    meta['visualization_definition']='Combined 1%-99% range, shared across frames/models; prediction panels use only common valid pixels.'
    cols=4 if reference else 3
    overview,axes=plt.subplots(a.frames,cols,figsize=(5*cols,4*a.frames),squeeze=False,layout='constrained')
    for i,(folder,rgb,sparse,depth,dmdc) in enumerate(prepared):
        common=np.isfinite(depth) if dmdc is None else np.isfinite(depth)&np.isfinite(dmdc)&(dmdc>0)
        panels=[('RGB',rgb),('Sparse camera Z',sparse),('LDCM (common valid)',np.where(common,depth,np.nan))]
        if dmdc is not None:
            panels.append(('DMDC (common valid)',np.where(common,dmdc,np.nan)))
        fig,row=plt.subplots(1,cols,figsize=(5*cols,4),layout='constrained')
        for targets in [row,axes[i]]:
            for ax,(title,data) in zip(targets,panels):
                if data.ndim==3:
                    ax.imshow(data)
                else:
                    artist=ax.imshow(np.ma.masked_invalid(data),cmap='turbo',vmin=lo,vmax=hi)
                ax.set_title(title); ax.axis('off')
        fig.colorbar(row[-1].images[0],ax=row,label='Depth (m)',shrink=.8)
        fig.savefig(folder/'comparison.png',dpi=120); plt.close(fig)
        for name,data in [('depth_vis',depth),('sparse_depth_vis',sparse)]:
            plt.imsave(folder/f'{name}.png',np.ma.masked_invalid(data),cmap='turbo',vmin=lo,vmax=hi)
    overview.colorbar(artist,ax=axes,label='Depth (m)',shrink=.8)
    overview.savefig(a.output/'overview.png',dpi=120); plt.close(overview)
    (a.output/'validation.json').write_text(json.dumps(dict(passed=True,frames=validations),indent=2))
    save()
    print(f'Saved {a.output}',flush=True)


if __name__=='__main__':
    main()
