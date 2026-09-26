"""WAV import (PCM 8/16/24/32-bit and IEEE float) using ``scipy.io.wavfile``.

Samples become float64 relative to digital full scale (unit ``FS``): signed PCM is divided by
2**(bits-1), unsigned 8-bit PCM is offset by 128 first, float data is kept as stored. A WAV file
carries no physical units, so no volt calibration is implied.

Channels are named ``ch1``..``chN`` (stereo: ``ch1`` = left, ``ch2`` = right). They are never
mixed down or dropped; analyses must select one explicitly when there is more than one.
"""
from __future__ import annotations

import io
import struct
import warnings
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from ..errors import DataImportError
from .dataset import Column, Dataset, SourceInfo

FULL_SCALE_UNIT = "FS"
_FORMAT_NAMES = {1: "PCM", 3: "IEEE float", 6: "A-law", 7: "mu-law"}


def load_wav(path: str | Path) -> Dataset:
    """Load a WAV file; the sample rate and format details are exposed in the dataset."""
    path = Path(path)
    source, raw = SourceInfo.read(path, "wav")
    header = _read_riff_chunks(raw, path.name)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rate, data = wavfile.read(io.BytesIO(raw), mmap=False)
    except Exception as exc:  # the parser raises assorted errors for malformed files
        raise DataImportError(f"{path.name}: cannot read WAV data ({exc})") from exc
    if rate <= 0:
        raise DataImportError(f"{path.name}: invalid sample rate {rate} Hz in the WAV header")
    if data.size == 0:
        raise DataImportError(f"{path.name}: the WAV file contains no samples")

    samples = _to_full_scale(data, path.name)
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]
    frames, channels = samples.shape
    columns = tuple(
        Column(f"ch{k + 1}", np.ascontiguousarray(samples[:, k]), FULL_SCALE_UNIT) for k in range(channels)
    )
    metadata = {
        "format": "wav",
        "sample_rate_hz": float(rate),
        "channels": channels,
        "channel_layout": _layout(channels),
        "frames": frames,
        "duration_s": frames / float(rate),
        "encoding": _encoding(header, data.dtype),
        "sample_dtype": str(data.dtype),
        "amplitude_unit": "FS (fraction of digital full scale; not calibrated to physical units)",
        "reader_warnings": [str(w.message) for w in caught],
        "comments": [header["info"]["ICMT"]] if header.get("info", {}).get("ICMT") else [],
        "info": header.get("info", {}),
    }
    # Reader warnings can mean damaged data (e.g. a truncated recording), so they are surfaced.
    notices = [f"{path.name}: {message}" for message in metadata["reader_warnings"]]
    declared = header.get("data_bytes", 0) // header["block_align"] if header.get("block_align") else 0
    if not header.get("rf64") and declared > frames:
        notices.append(f"{path.name}: the data chunk declares {declared} frames but the file holds only {frames}; "
                       "the recording is truncated")
    return Dataset(source=source, columns=columns, sample_rate_hz=float(rate), metadata=metadata,
                   warnings=tuple(notices))


def _read_riff_chunks(raw: bytes, name: str) -> dict:
    """Check the RIFF/WAVE signature; return format details, the declared data size and INFO text."""
    if len(raw) < 12 or raw[:4] not in (b"RIFF", b"RIFX", b"RF64") or raw[8:12] != b"WAVE":
        raise DataImportError(f"{name}: not a WAV file (missing RIFF/WAVE header)", hint="format")
    endian = ">" if raw[:4] == b"RIFX" else "<"
    found: dict = {"rf64": raw[:4] == b"RF64"}  # RF64 keeps real sizes in a ds64 chunk: no size check
    pos = 12
    while pos + 8 <= len(raw):
        chunk_id, size = raw[pos:pos + 4], struct.unpack(endian + "I", raw[pos + 4:pos + 8])[0]
        body = raw[pos + 8:pos + 8 + size]
        if chunk_id == b"fmt " and len(body) >= 16:
            tag, _channels, _rate, _byte_rate, align, bits = struct.unpack(endian + "HHIIHH", body[:16])
            if tag == 0xFFFE and len(body) >= 26:  # WAVE_FORMAT_EXTENSIBLE: use the sub-format
                tag = struct.unpack(endian + "H", body[24:26])[0]
            found.update(format_tag=tag, bits_per_sample=bits, block_align=align)
        elif chunk_id == b"data":
            found["data_bytes"] = size
        elif chunk_id == b"LIST" and body[:4] == b"INFO":
            found["info"] = _parse_info(body[4:], endian)
        pos += 8 + size + (size & 1)
    return found


def _parse_info(body: bytes, endian: str) -> dict[str, str]:
    """Text sub-chunks of a RIFF LIST/INFO chunk, e.g. ICMT (comment) or ISFT (software)."""
    info: dict[str, str] = {}
    pos = 0
    while pos + 8 <= len(body):
        key = body[pos:pos + 4].decode("ascii", "replace")
        size = struct.unpack(endian + "I", body[pos + 4:pos + 8])[0]
        info[key] = body[pos + 8:pos + 8 + size].split(b"\x00", 1)[0].decode("utf-8", "replace").strip()
        pos += 8 + size + (size & 1)
    return info


def _to_full_scale(data: np.ndarray, name: str) -> np.ndarray:
    if data.dtype == np.uint8:
        return (data.astype(np.float64) - 128.0) / 128.0
    if np.issubdtype(data.dtype, np.signedinteger):
        # scipy returns 24-bit (and other odd widths) left-justified in the next wider type.
        return data.astype(np.float64) / float(2 ** (8 * data.dtype.itemsize - 1))
    if np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float64)
    raise DataImportError(f"{name}: unsupported WAV sample type {data.dtype}")


def _encoding(header: dict[str, int], dtype: np.dtype) -> str:
    if "format_tag" in header:
        label = _FORMAT_NAMES.get(header["format_tag"], f"format tag {header['format_tag']}")
        return f"{label} {header['bits_per_sample']}-bit"
    return f"{dtype} samples"


def _layout(channels: int) -> str:
    if channels == 1:
        return "mono"
    if channels == 2:
        return "stereo (ch1 = left, ch2 = right)"
    return f"{channels} channels"
