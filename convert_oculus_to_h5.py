#!/usr/bin/env python3
# convert_oculus_to_h5.py
#
# Usage:
#   python3 convert_oculus_to_h5.py input.oculus output.h5
#
# Requires:
#   pip install h5py numpy

import sqlite3
import struct
import zlib
import sys
from pathlib import Path

import h5py
import numpy as np


def decompress_payload(payload: bytes) -> bytes:
    """Blueprint .oculus logging payload:
       uint32 decompressed_size + zlib-compressed Oculus message.
    """
    expected_size = struct.unpack_from("<I", payload, 0)[0]
    raw = zlib.decompress(payload[4:])
    if len(raw) != expected_size:
        raise ValueError(f"decompressed size mismatch: {len(raw)} != {expected_size}")
    return raw


def parse_simple_fire_response(raw: bytes):
    """Parse Oculus Simple Fire V1/V2 response.
       Returns metadata dict and image array [range, bearing].
    """
    oculus_id = struct.unpack_from("<H", raw, 0)[0]
    message_id = struct.unpack_from("<H", raw, 6)[0]
    version = struct.unpack_from("<H", raw, 8)[0]

    if oculus_id != 0x4F53:
        raise ValueError("Not an Oculus message")
    if message_id != 35:
        raise ValueError(f"Not SimpleFire response: message_id={message_id}")

    # V2 response. In the uploaded M3000d log, version is 2.
    if version == 2:
        ping_id = struct.unpack_from("<I", raw, 89)[0]
        frequency = struct.unpack_from("<d", raw, 97)[0]
        temperature = struct.unpack_from("<d", raw, 105)[0]
        pressure = struct.unpack_from("<d", raw, 113)[0]
        heading = struct.unpack_from("<d", raw, 121)[0]
        pitch = struct.unpack_from("<d", raw, 129)[0]
        roll = struct.unpack_from("<d", raw, 137)[0]
        speed_of_sound = struct.unpack_from("<d", raw, 145)[0]
        ping_start_time = struct.unpack_from("<d", raw, 153)[0]
        data_size = struct.unpack_from("<B", raw, 161)[0]
        range_resolution = struct.unpack_from("<d", raw, 162)[0]
        range_count = struct.unpack_from("<H", raw, 170)[0]
        bearing_count = struct.unpack_from("<H", raw, 172)[0]
        image_offset = struct.unpack_from("<I", raw, 190)[0]
        image_size = struct.unpack_from("<I", raw, 194)[0]
        message_size = struct.unpack_from("<I", raw, 198)[0]
        bearings_offset = 202

    # V1 response.
    elif version == 1:
        ping_id = struct.unpack_from("<I", raw, 53)[0]
        frequency = struct.unpack_from("<d", raw, 61)[0]
        temperature = struct.unpack_from("<d", raw, 69)[0]
        pressure = struct.unpack_from("<d", raw, 77)[0]
        heading = np.nan
        pitch = np.nan
        roll = np.nan
        speed_of_sound = struct.unpack_from("<d", raw, 85)[0]
        ping_start_time = float(struct.unpack_from("<I", raw, 93)[0])
        data_size = struct.unpack_from("<B", raw, 97)[0]
        range_resolution = struct.unpack_from("<d", raw, 98)[0]
        range_count = struct.unpack_from("<H", raw, 106)[0]
        bearing_count = struct.unpack_from("<H", raw, 108)[0]
        image_offset = struct.unpack_from("<I", raw, 110)[0]
        image_size = struct.unpack_from("<I", raw, 114)[0]
        message_size = struct.unpack_from("<I", raw, 118)[0]
        bearings_offset = 122
    else:
        raise ValueError(f"Unsupported SimpleFire version: {version}")

    # bearings are int16 values. Usually centi-degrees; divide by 100 for degrees.
    bearings_raw = np.frombuffer(
        raw[bearings_offset:bearings_offset + 2 * bearing_count],
        dtype="<i2"
    ).astype(np.float32)
    bearings_deg = bearings_raw / 100.0

    if data_size == 0:
        dtype = np.uint8
        bytes_per_pixel = 1
    elif data_size == 1:
        dtype = "<u2"
        bytes_per_pixel = 2
    elif data_size == 2:
        # uncommon; store raw bytes if 24-bit
        dtype = np.uint8
        bytes_per_pixel = 3
    elif data_size == 3:
        dtype = "<u4"
        bytes_per_pixel = 4
    else:
        raise ValueError(f"Unsupported data_size: {data_size}")

    expected_image_size = range_count * bearing_count * bytes_per_pixel
    if image_size < expected_image_size:
        raise ValueError(f"imageSize too small: {image_size} < {expected_image_size}")

    image_bytes = raw[image_offset:image_offset + expected_image_size]

    if data_size == 2:
        image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(range_count, bearing_count, 3)
    else:
        image = np.frombuffer(image_bytes, dtype=dtype).reshape(range_count, bearing_count)

    meta = {
        "version": version,
        "ping_id": ping_id,
        "frequency": frequency,
        "temperature": temperature,
        "pressure": pressure,
        "heading": heading,
        "pitch": pitch,
        "roll": roll,
        "speed_of_sound": speed_of_sound,
        "ping_start_time": ping_start_time,
        "data_size": data_size,
        "range_resolution": range_resolution,
        "range_count": range_count,
        "bearing_count": bearing_count,
        "image_offset": image_offset,
        "image_size": image_size,
        "message_size": message_size,
        "bearings_deg": bearings_deg,
    }
    return meta, image


def convert_oculus_to_h5(input_oculus: str, output_h5: str):
    con = sqlite3.connect(input_oculus)
    cur = con.cursor()

    rows = cur.execute(
        "SELECT entryId, timestamp, payload FROM data ORDER BY entryId"
    ).fetchall()

    if not rows:
        raise RuntimeError("No rows in data table")

    # First pass: parse frames and get max sizes.
    parsed = []
    max_range = 0
    max_bearing = 0

    for entry_id, timestamp, payload in rows:
        raw = decompress_payload(payload)
        try:
            meta, image = parse_simple_fire_response(raw)
        except ValueError:
            continue

        parsed.append((entry_id, timestamp, meta, image))
        max_range = max(max_range, meta["range_count"])
        max_bearing = max(max_bearing, meta["bearing_count"])

    n = len(parsed)
    if n == 0:
        raise RuntimeError("No SimpleFire response frames found")

    # Assume uint8 or uint16 image. Use dtype from first image.
    image_dtype = parsed[0][3].dtype

    with h5py.File(output_h5, "w") as h5:
        h5.attrs["source_file"] = str(input_oculus)
        h5.attrs["description"] = "Converted from Blueprint/Oculus .oculus SQLite log"

        images = h5.create_dataset(
            "/images",
            shape=(max_range, max_bearing, n),
            dtype=image_dtype,
            compression="gzip",
            compression_opts=4,
            chunks=(max_range, max_bearing, 1),
        )

        entry_id_arr = np.zeros(n, dtype=np.int64)
        timestamp_arr = np.zeros(n, dtype=np.int64)
        ping_id_arr = np.zeros(n, dtype=np.uint32)
        frequency_arr = np.zeros(n, dtype=np.float64)
        temperature_arr = np.zeros(n, dtype=np.float64)
        pressure_arr = np.zeros(n, dtype=np.float64)
        heading_arr = np.zeros(n, dtype=np.float64)
        pitch_arr = np.zeros(n, dtype=np.float64)
        roll_arr = np.zeros(n, dtype=np.float64)
        sos_arr = np.zeros(n, dtype=np.float64)
        ping_time_arr = np.zeros(n, dtype=np.float64)
        data_size_arr = np.zeros(n, dtype=np.uint8)
        range_resolution_arr = np.zeros(n, dtype=np.float64)
        range_count_arr = np.zeros(n, dtype=np.uint16)
        bearing_count_arr = np.zeros(n, dtype=np.uint16)
        bearings = np.full((max_bearing, n), np.nan, dtype=np.float32)

        for k, (entry_id, timestamp, meta, image) in enumerate(parsed):
            rc = meta["range_count"]
            bc = meta["bearing_count"]
            images[0:rc, 0:bc, k] = image

            entry_id_arr[k] = entry_id
            timestamp_arr[k] = timestamp
            ping_id_arr[k] = meta["ping_id"]
            frequency_arr[k] = meta["frequency"]
            temperature_arr[k] = meta["temperature"]
            pressure_arr[k] = meta["pressure"]
            heading_arr[k] = meta["heading"]
            pitch_arr[k] = meta["pitch"]
            roll_arr[k] = meta["roll"]
            sos_arr[k] = meta["speed_of_sound"]
            ping_time_arr[k] = meta["ping_start_time"]
            data_size_arr[k] = meta["data_size"]
            range_resolution_arr[k] = meta["range_resolution"]
            range_count_arr[k] = rc
            bearing_count_arr[k] = bc
            bearings[0:bc, k] = meta["bearings_deg"]

        h5.create_dataset("/entryId", data=entry_id_arr)
        h5.create_dataset("/timestamp", data=timestamp_arr)
        h5.create_dataset("/pingId", data=ping_id_arr)
        h5.create_dataset("/frequency", data=frequency_arr)
        h5.create_dataset("/temperature", data=temperature_arr)
        h5.create_dataset("/pressure", data=pressure_arr)
        h5.create_dataset("/heading", data=heading_arr)
        h5.create_dataset("/pitch", data=pitch_arr)
        h5.create_dataset("/roll", data=roll_arr)
        h5.create_dataset("/speedOfSound", data=sos_arr)
        h5.create_dataset("/pingStartTime", data=ping_time_arr)
        h5.create_dataset("/dataSize", data=data_size_arr)
        h5.create_dataset("/rangeResolution", data=range_resolution_arr)
        h5.create_dataset("/rangeCount", data=range_count_arr)
        h5.create_dataset("/bearingCount", data=bearing_count_arr)
        h5.create_dataset("/bearingsDeg", data=bearings)

    con.close()
    print(f"Converted {n} frames")
    print(f"Output: {output_h5}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 convert_oculus_to_h5.py input.oculus output.h5")
        sys.exit(1)

    convert_oculus_to_h5(sys.argv[1], sys.argv[2])
