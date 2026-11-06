#!/usr/bin/env python3
"""Torch-free extraction of the generator weights from a Coqui TTS training checkpoint (.pth).

Reads the zip-format torch checkpoint with a restricted unpickler (no torch needed, nothing
from the file is executed), keeps checkpoint['model'] entries except the discriminator
('disc.*'), and writes them as float32 .npy arrays inside .npz parts of < 95 MB each, plus a
manifest with sha256 sums.

usage: python3 -I strip_pth.py CHECKPOINT.pth OUTDIR
"""
import collections, hashlib, io, json, os, pickle, sys, zipfile
import numpy as np

DT = {'FloatStorage': np.float32, 'HalfStorage': np.float16, 'DoubleStorage': np.float64,
      'LongStorage': np.int64, 'IntStorage': np.int32, 'ShortStorage': np.int16,
      'CharStorage': np.int8, 'ByteStorage': np.uint8, 'BoolStorage': np.bool_}
DTYPE_NAMES = {'float32': np.float32, 'float': np.float32, 'float16': np.float16, 'half': np.float16,
               'float64': np.float64, 'double': np.float64, 'int64': np.int64, 'long': np.int64,
               'int32': np.int32, 'int': np.int32, 'int16': np.int16, 'int8': np.int8,
               'uint8': np.uint8, 'bool': np.bool_}


class Dummy:
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): pass
    def __call__(self, *a, **k): return Dummy()
    def __setitem__(self, k, v): pass
    def append(self, v): pass
    def extend(self, v): pass


class DType:
    def __init__(self, name): self.name = name


class LazyStorage:
    def __init__(self, zf, prefix, key, dtype):
        self.zf, self.prefix, self.key, self.dtype = zf, prefix, key, dtype
    def array(self):
        return np.frombuffer(self.zf.read(f'{self.prefix}/data/{self.key}'), dtype=self.dtype)


class LazyTensor:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)
    def array(self):
        buf = self.storage.array()
        item = buf.itemsize
        if len(self.size) == 0:
            return np.array(buf[self.offset])
        view = np.lib.stride_tricks.as_strided(buf[self.offset:], shape=self.size,
                                               strides=[s * item for s in self.stride])
        return np.array(view)


def rebuild_tensor(storage, offset, size, stride, *rest):
    return LazyTensor(storage, offset, size, stride)


def rebuild_parameter(data, *rest):
    return data


def load(path):
    zf = zipfile.ZipFile(path)
    pkl = [n for n in zf.namelist() if n.endswith('/data.pkl')][0]
    prefix = pkl[:-len('/data.pkl')]

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == 'torch._utils' and name == '_rebuild_tensor_v2':
                return rebuild_tensor
            if mod == 'torch._utils' and name == '_rebuild_parameter':
                return rebuild_parameter
            if mod == 'torch' and name.endswith('Storage'):
                return ('storage', name)
            if mod == 'torch' and name in DTYPE_NAMES:
                return DType(name)
            if mod == 'collections' and name == 'OrderedDict':
                return collections.OrderedDict
            if mod == 'builtins' and name in ('set', 'frozenset', 'dict', 'list', 'tuple', 'slice'):
                return getattr(__import__('builtins'), name)
            return Dummy

        def persistent_load(self, pid):
            _, stype, key, _loc, _numel = pid
            if isinstance(stype, tuple):
                dtype = DT[stype[1]]
            elif isinstance(stype, DType):
                dtype = DTYPE_NAMES[stype.name]
            else:
                dtype = np.uint8
            return LazyStorage(zf, prefix, key, dtype)

    return U(io.BytesIO(zf.read(pkl))).load()


def main(src, outdir):
    os.makedirs(outdir, exist_ok=True)
    ck = load(src)
    sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    keep = collections.OrderedDict()
    for k, v in sd.items():
        if k.startswith('disc.') or not isinstance(v, LazyTensor):
            continue
        a = v.array()
        if a.dtype in (np.float16, np.float64):
            a = a.astype(np.float32)
        keep[k] = a
    total = sum(a.nbytes for a in keep.values())
    print(f'kept {len(keep)} tensors, {total/1e6:.1f} MB (skipped discriminator/optimizer)')
    parts, cur, size = [], {}, 0
    for k, a in keep.items():
        if cur and size + a.nbytes > 90e6:
            parts.append(cur); cur, size = {}, 0
        cur[k] = a; size += a.nbytes
    if cur:
        parts.append(cur)
    man = {'source': os.path.basename(src), 'source_sha256': sha(src), 'parts': []}
    for i, p in enumerate(parts):
        fn = os.path.join(outdir, f'weights.part{i}.npz')
        np.savez(fn, **{k.replace('/', '|'): v for k, v in p.items()})
        man['parts'].append({'file': os.path.basename(fn), 'sha256': sha(fn), 'bytes': os.path.getsize(fn), 'tensors': len(p)})
    json.dump(man, open(os.path.join(outdir, 'manifest.json'), 'w'), indent=1)
    print(json.dumps(man, indent=1))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
