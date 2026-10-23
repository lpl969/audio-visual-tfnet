#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import numpy as np
import spaudiopy

MIC_POSITIONS = np.array(
    [
        [0.4, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [-0.4, 0.0, 0.0],
        [0.0, -0.4, 0.0],
    ],
    dtype=np.float32,
)


def concat_audio(audio_path, file_name):
    audio_data = np.load(os.path.join(audio_path, file_name))
    audio_data = np.asarray(audio_data, dtype=np.float32)
    if audio_data.ndim != 2:
        raise ValueError(f"Expected audio array with 2 dims, got shape={audio_data.shape} for {file_name}")
    if audio_data.shape[0] != 4 and audio_data.shape[-1] == 4:
        audio_data = audio_data.T
    if audio_data.shape[0] != 4 or not np.isfinite(audio_data).all():
        raise ValueError("Audio must contain four finite channels")
    return audio_data


def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(y, x)
    col = np.arccos(z / (r + 1e-8))
    return r, az, col


def microphone_positions_spherical(mic_positions):
    sph_coords = np.zeros_like(mic_positions)
    for i, (x, y, z) in enumerate(mic_positions):
        sph_coords[i] = cart2sph(x, y, z)
    return sph_coords


def compute_sh_features(audio, order=4, sh_type='real'):
    mic_pos = MIC_POSITIONS
    sph = microphone_positions_spherical(mic_pos)
    _, az, col = sph[:, 0], sph[:, 1], sph[:, 2]
    shc = spaudiopy.sph.src_to_sh(audio, az, col, order, sh_type)
    return shc.astype(np.float32)
