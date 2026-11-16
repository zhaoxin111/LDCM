"""Odin1 bag, FishPoly projection, statistics and PLY helpers.

Copied from dToF-DMDC infer_odin1.py for standalone LDCM testing.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

IMAGE_TOPIC = '/odin1/image/compressed'
CLOUD_TOPIC = '/odin1/cloud_raw'
CONSISTENCY_DEPTH_EDGES_M = (0., 5., 10., 15., 20., float('inf'))


def stamp_ns(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def read_samples(bag, count, max_delta):
    """Two passes keep only selected messages in memory; use sensor header times."""
    # Older ROS2 bags omit message definitions. The standard image/cloud types
    # used here have the same definitions in Humble; embedded definitions win.
    with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        connections = [c for c in reader.connections if c.topic in (IMAGE_TOPIC, CLOUD_TOPIC)]
        times = {IMAGE_TOPIC: [], CLOUD_TOPIC: []}
        for c, _, data in reader.messages(connections=connections):
            times[c.topic].append(stamp_ns(reader.deserialize(data, c.msgtype)))
        if any(not t for t in times.values()):
            raise ValueError('Bag must contain compressed RGB and raw point clouds')
        images = np.array(times[IMAGE_TOPIC], dtype=np.int64)
        clouds = np.array(times[CLOUD_TOPIC], dtype=np.int64)
        pairs, used = [], set()
        for fraction in np.linspace(.2, .8, count) if count > 1 else [.5]:
            target = images.min() + fraction * (images.max() - images.min())
            for index in np.argsort(np.abs(images - target)):
                it = int(images[index])
                ct = int(clouds[np.argmin(np.abs(clouds - it))])
                if it not in used and abs(ct - it) <= max_delta * 1e9:
                    pairs.append((it, ct, float(fraction)))
                    used.add(it)
                    break
            else:
                raise ValueError('Cannot find enough distinct synchronized frames')
        wanted = {IMAGE_TOPIC: {p[0] for p in pairs}, CLOUD_TOPIC: {p[1] for p in pairs}}
        selected = {}
        for c, _, data in reader.messages(connections=connections):
            msg = reader.deserialize(data, c.msgtype)
            t = stamp_ns(msg)
            if t in wanted[c.topic]:
                selected[c.topic, t] = msg
        return [(selected[IMAGE_TOPIC, it], selected[CLOUD_TOPIC, ct], fraction) for it, ct, fraction in pairs]


def rectify_maps(cam, width, height, fov):
    f = width / (2 * np.tan(np.deg2rad(fov) / 2))
    k = np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1]])
    v, u = np.mgrid[:height, :width]
    x, y = (u - k[0, 2]) / f, (v - k[1, 2]) / f
    r = np.hypot(x, y)
    theta = np.arctan(r)
    td = theta.copy()
    for power in range(2, 8):
        td += cam[f'k{power}'] * theta ** power
    scale = np.divide(td, r, out=np.ones_like(r), where=r > 1e-12)
    xd, yd = x * scale, y * scale
    mx = cam['A11'] * xd + cam['A12'] * yd + cam['u0']
    my = cam['A22'] * yd + cam['v0']
    valid = (mx >= 0) & (mx < cam['image_width'] - 1) & (my >= 0) & (my < cam['image_height'] - 1)
    return k, mx.astype(np.float32), my.astype(np.float32), valid


def project_cloud(msg, transform, k, width, height, confidence, min_range=0.1):
    formats = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
    endian = '>' if msg.is_bigendian else '<'
    fields = {f.name: f for f in msg.fields}
    names = ['x', 'y', 'z', 'confidence']
    dt = np.dtype({'names': names, 'formats': [endian + formats[fields[n].datatype] for n in names],
                   'offsets': [fields[n].offset for n in names], 'itemsize': msg.point_step})
    p = np.ndarray((msg.height, msg.width), dtype=dt, buffer=msg.data, strides=(msg.row_step, msg.point_step)).reshape(-1)
    xyz = np.stack([p[n] for n in names[:3]], axis=1).astype(np.float64)
    ranges = np.linalg.norm(xyz, axis=1)
    confidence_valid = np.isfinite(xyz).all(axis=1) & (ranges > 0) & (p['confidence'] >= confidence)
    # Confidence alone does not reject millimeter-scale false returns in Odin1 bags.
    # Filter in the LiDAR frame before the camera translation can move them into view.
    good = confidence_valid & (ranges >= min_range)
    xyz = xyz[good] @ transform[:3, :3].T + transform[:3, 3]
    xyz = xyz[xyz[:, 2] > 0]
    uv = xyz @ k.T
    uv = uv[:, :2] / xyz[:, 2:3]
    inside = (uv[:, 0] >= -.5) & (uv[:, 0] < width - .5) & (uv[:, 1] >= -.5) & (uv[:, 1] < height - .5)
    uv = np.floor(uv[inside] + .5).astype(int)
    z = xyz[inside, 2]
    depth = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(depth, uv[:, 1] * width + uv[:, 0], z)
    depth[~np.isfinite(depth)] = np.nan
    stats = {'filtered_input_points': int(good.sum()),
             'rejected_near_range_points': int((confidence_valid & (ranges < min_range)).sum())}
    if 'offset_time' in fields:
        field = fields['offset_time']
        offsets = np.ndarray((msg.height, msg.width), dtype=endian + formats[field.datatype],
                             buffer=msg.data, offset=field.offset,
                             strides=(msg.row_step, msg.point_step)).reshape(-1)[good]
        if np.isfinite(offsets).all() and len(offsets):
            stats['point_time_offset_range_ms'] = [float(offsets.min() * 1000), float(offsets.max() * 1000)]
    return depth.reshape(height, width), stats


def depth_binned_consistency(depth, sparse, valid):
    """Compare shared pixels in [lower, upper) bins of sparse camera-Z depth.

    Relative errors are ratios (0.05 means 5%); positive signed bias means
    predicted depth is farther away. An upper bound of None means unbounded.
    Empty bins retain their bounds/count and have null error metrics in JSON.
    Sparse counts/fractions include valid sparse pixels even without predictions.
    """
    sparse_valid = np.isfinite(sparse) & (sparse > 0)
    sparse_reference = sparse[sparse_valid]
    total_sparse = int(sparse_valid.sum())
    shared = valid & np.isfinite(depth) & (depth > 0) & sparse_valid
    reference = sparse[shared].astype(np.float64)
    error = depth[shared].astype(np.float64) - reference
    results = []
    for lower, upper in zip(CONSISTENCY_DEPTH_EDGES_M[:-1], CONSISTENCY_DEPTH_EDGES_M[1:]):
        selected = (reference >= lower) & (reference < upper)
        count = int(selected.sum())
        sparse_count = int(((sparse_reference >= lower) & (sparse_reference < upper)).sum())
        metrics = dict.fromkeys(['sparse_mae_m', 'sparse_abs_relative_error',
                                 'sparse_signed_relative_bias', 'sparse_abs_relative_error_median',
                                 'sparse_abs_relative_error_p90', 'sparse_relative_error_gt_5pct_fraction',
                                 'sparse_relative_error_gt_10pct_fraction'])
        if count:
            relative = error[selected] / reference[selected]
            absolute_relative = np.abs(relative)
            metrics.update(
                sparse_mae_m=float(np.abs(error[selected]).mean()),
                sparse_abs_relative_error=float(absolute_relative.mean()),
                sparse_signed_relative_bias=float(relative.mean()),
                sparse_abs_relative_error_median=float(np.median(absolute_relative)),
                sparse_abs_relative_error_p90=float(np.percentile(absolute_relative, 90)),
                sparse_relative_error_gt_5pct_fraction=float((absolute_relative > .05).mean()),
                sparse_relative_error_gt_10pct_fraction=float((absolute_relative > .10).mean()),
            )
        results.append(dict(depth_min_m=lower, depth_max_m=upper if np.isfinite(upper) else None,
                            sparse_pixels=sparse_count,
                            sparse_pixel_fraction=sparse_count / total_sparse if total_sparse else None,
                            consistency_pixels=count, **metrics))
    return results


def write_ply(path, depth, rgb, k, valid):
    v, u = np.nonzero(valid)
    z = depth[v, u]
    points = np.stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z], axis=1)
    vertices = np.empty(len(z), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for i, key in enumerate(['x', 'y', 'z']):
        vertices[key] = points[:, i]
    for i, key in enumerate(['red', 'green', 'blue']):
        vertices[key] = rgb[v, u, i]
    header = f'ply\nformat binary_little_endian 1.0\ncomment camera optical frame; meters\nelement vertex {len(z)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'
    with path.open('wb') as f:
        f.write(header.encode('ascii'))
        vertices.tofile(f)
    return len(z)
