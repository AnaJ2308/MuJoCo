#!/usr/bin/env python3
"""
Laptop HField Viewer (TCP -> MuJoCo)

- Connects to the Jetson server over TCP
- Receives normalized heightfield grids (0..1)
- Updates a MuJoCo heightfield live
"""

import socket, struct, zlib
import sys, time
import numpy as np
import mujoco
from mujoco import viewer

MAGIC = b'HFLD'

def recv_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('socket closed')
        buf.extend(chunk)
    return bytes(buf)

def run_viewer(jetson_host: str, port: int=5005, L: float=6.0, Hz: float=0.5, base: float=0.001):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((jetson_host, port))
    print(f'Connected to {jetson_host}:{port}')

    # Wait for first frame, then build model and start viewer
    header_fmt = '<4sBHHIffI'
    header_size = struct.calcsize(header_fmt)

    
    # First packet (RAW METERS)
    header = recv_exact(s, header_size)
    magic, version, nrow, ncol, seq, zmin, zmax, paylen = struct.unpack(header_fmt, header)
    if magic != MAGIC: raise ValueError('Bad magic')
    raw = zlib.decompress(recv_exact(s, paylen))
    Hm = np.frombuffer(raw, dtype=np.float32).reshape((nrow, ncol))  # meters

    # Your display mapping (meters -> [0,1])
    FLOOR = 0.20      # meters you want to be "flat ground"
    HEAD  = 2.0       # vertical headroom; raise if you never want to clip
    BASE_PLATE = 0.0001  # thin slab so no visible rim

    # TODO: np.clip does not normalize, it constrins you to defined bounds. and add this outside of this function
    def to_h01_meters(Hm_):
        return np.clip((Hm_ - FLOOR) / HEAD, 0.0, 1.0).astype(np.float32, copy=False)

    H01 = to_h01_meters(Hm)

    # Build MuJoCo model
    Hz_use   = HEAD         # 1.0 in hfield == HEAD meters above FLOOR
    base_use = BASE_PLATE   # thin base
    print(f"[viewer] floor={FLOOR:.3f} m, headroom={HEAD:.3f} m, base={base_use:.3f} m")

    xml = f'''
<mujoco model="hf_viewer">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <asset><hfield name="terrain" nrow="{nrow}" ncol="{ncol}" size="{L} {L} {Hz_use} {base_use}"/></asset>
  <worldbody>
    <geom type="hfield" hfield="terrain" pos="0 0 0" rgba="0.7 0.85 0.7 1"/>
    <body pos="0 0 1.0"><freejoint/><geom type="sphere" size="0.08" density="2000" rgba="0.9 0.2 0.2 1"/></body>
  </worldbody>
</mujoco>'''

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, 'terrain')

    # Write first frame
    adr = model.hfield_adr[hid]
    model.hfield_data[adr: adr + nrow*ncol] = H01.ravel(order='C')

    with viewer.launch_passive(model, data) as v:
        v.cam.lookat[:] = (0.0, 0.0, 0.3)
        v.cam.azimuth = 90
        v.cam.elevation = -20
        v.cam.distance = 7.0

        # Ensure first upload
        if hasattr(v, 'update_hfield'): v.update_hfield(hid)
        else:
            try: mujoco.mjr_uploadHField(model, hid, v.context)
            except Exception: pass

        while v.is_running():
            # Get next packet
            header = recv_exact(s, header_size)
            magic, version, nrow, ncol, seq, zmin, zmax, paylen = struct.unpack(header_fmt, header)
            if magic != MAGIC: raise ValueError('Bad magic')
            raw = zlib.decompress(recv_exact(s, paylen))
            Hm = np.frombuffer(raw, dtype=np.float32).reshape((nrow, ncol))  # meters

            H01 = to_h01_meters(Hm)

            model.hfield_data[adr: adr + nrow*ncol] = H01.ravel(order='C')
            if hasattr(v, 'update_hfield'): v.update_hfield(hid)
            else:
                try: mujoco.mjr_uploadHField(model, hid, v.context)
                except Exception: pass

            mujoco.mj_step(model, data)
            v.sync()

if __name__ == '__main__':
    # Usage: python3 laptop_hfield_viewer.py <JETSON_IP> [port]
    ip = sys.argv[1] if len(sys.argv) > 1 else '127.0.0.1'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5005
    run_viewer(ip, port)
