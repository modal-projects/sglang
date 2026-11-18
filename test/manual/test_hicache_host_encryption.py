import ctypes, importlib.util, os, random, sys, threading, time, types
import numpy as np, torch

SRT = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "python/sglang/srt")
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod
for pkg in ("sglang", "sglang.srt", "sglang.srt.mem_cache", "sglang.srt.mem_cache.pool_host"):
    sys.modules[pkg] = types.ModuleType(pkg)
load("sglang.srt.environ", f"{SRT}/environ.py")
base = types.ModuleType("sglang.srt.mem_cache.pool_host.base")
base.uses_shared_host_layout = lambda pool: False
sys.modules["sglang.srt.mem_cache.pool_host.base"] = base
he = load("sglang.srt.mem_cache.host_encryption", f"{SRT}/mem_cache/host_encryption.py")
from sglang.srt.environ import envs

class Pool:
    """K block then V block, page-first, like the MHA host pool."""
    def __init__(self, pages, page_size, token_bytes):
        self.page_size, self.token_bytes, self.pages = page_size, token_bytes, pages
        self.size = pages * page_size
        self.half = pages * page_size * token_bytes
        self.buf = np.zeros(2 * self.half, dtype=np.uint8)
        self.base = self.buf.ctypes.data
    def get_page_buffer_meta(self, indices):
        ptrs = []
        for i in range(0, len(indices), self.page_size):
            k = self.base + int(indices[i]) * self.token_bytes
            ptrs += [k, k + self.half]
        return ptrs, [self.page_size * self.token_bytes] * len(ptrs)
    def view(self, page):
        n = self.page_size * self.token_bytes
        return self.buf[page * n:(page + 1) * n], self.buf[self.half + page * n: self.half + (page + 1) * n]
    def read(self, page):
        k, v = self.view(page); return k.tobytes() + v.tobytes()
    def write(self, page, data):
        k, v = self.view(page); n = len(k)
        k[:] = np.frombuffer(data[:n], np.uint8); v[:] = np.frombuffer(data[n:], np.uint8)

class T:
    def __init__(self, pool, pages):
        self.host_pool = pool
        self.host_indices = torch.cat([torch.arange(p * pool.page_size, (p + 1) * pool.page_size) for p in pages])
class Ev:
    def __init__(self, delay=0.0): self.delay = delay
    def synchronize(self): time.sleep(self.delay)

def wait_sealed(enc, n, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        s = enc.stats()
        if s["pages_sealed"] == n and s["pages_plaintext"] == 0: return
        time.sleep(0.005)
    raise AssertionError(f"not sealed: {enc.stats()}")

def main():
    assert he.HostPoolEncryption.maybe_create() is None
    with envs.SGLANG_ENABLE_HICACHE_HOST_ENCRYPTION.override(True):
        enc = he.HostPoolEncryption.maybe_create()
    pool = Pool(pages=64, page_size=16, token_bytes=4096)
    plain = {p: os.urandom(2 * 16 * 4096) for p in range(64)}

    # backup -> encrypted at rest
    pages = list(range(8))
    t = enc.begin_backup([T(pool, pages)])
    for p in pages: pool.write(p, plain[p])          # the "GPU kernel"
    enc.finish_backup(t, Ev(0.01)); wait_sealed(enc, 8)
    sealed = {p: pool.read(p) for p in pages}
    assert all(sealed[p] != plain[p] for p in pages)
    # no plaintext 64-byte window survives anywhere in the pool
    blob = pool.buf.tobytes()
    assert all(plain[p][o:o + 64] not in blob for p in pages for o in (0, 4096, 70000))

    # load -> plaintext during the load, sealed again after, under a fresh nonce
    tr = T(pool, [3, 1, 6]); t = enc.begin_load([tr, tr])
    assert all(pool.read(p) == plain[p] for p in (3, 1, 6))
    assert pool.read(0) == sealed[0]
    enc.finish_load(t, Ev(0.01)); wait_sealed(enc, 8)
    assert all(pool.read(p) not in (plain[p], sealed[p]) for p in (3, 1, 6))
    t = enc.begin_load([T(pool, [3])]); assert pool.read(3) == plain[3]
    enc.finish_load(t, Ev()); wait_sealed(enc, 8)

    # overlapping loads keep the page open until the last one finishes
    t1 = enc.begin_load([T(pool, [2])]); t2 = enc.begin_load([T(pool, [2])])
    enc.finish_load(t1, Ev()); time.sleep(0.1)
    assert pool.read(2) == plain[2] and enc.stats()["pages_plaintext"] == 1
    enc.finish_load(t2, Ev()); wait_sealed(enc, 8)

    # a load that arrives before the backup was sealed sees plaintext, then seals
    t = enc.begin_backup([T(pool, [10])]); pool.write(10, plain[10])
    tl = enc.begin_load([T(pool, [10])]); assert pool.read(10) == plain[10]
    enc.finish_backup(t, Ev()); time.sleep(0.1); assert pool.read(10) == plain[10]
    enc.finish_load(tl, Ev()); wait_sealed(enc, 9); assert pool.read(10) != plain[10]

    # slot reuse: a new backup replaces the page; the stale load ticket is ignored
    tl = enc.begin_load([T(pool, [5])])
    t = enc.begin_backup([T(pool, [5])]); new5 = os.urandom(len(plain[5])); pool.write(5, new5)
    enc.finish_load(tl, Ev()); enc.finish_backup(t, Ev(0.05)); wait_sealed(enc, 9)
    t = enc.begin_load([T(pool, [5])]); assert pool.read(5) == new5
    enc.finish_load(t, Ev()); wait_sealed(enc, 9)

    # tampering and moved pages are refused
    k, _ = pool.view(4); k[100] ^= 1
    try: enc.begin_load([T(pool, [4])]); raise SystemExit("tamper not detected")
    except he.HostEncryptionError as e: print("tamper refused:", e)
    pool.write(7, sealed[0])                         # replay page 0's ciphertext into slot 7
    try: enc.begin_load([T(pool, [7])]); raise SystemExit("replay not detected")
    except he.HostEncryptionError: print("replay refused")

    # misaligned / partial transfers are refused
    bad = T(pool, [20]); bad.host_indices = bad.host_indices[:8]
    try: enc.begin_backup([bad]); raise SystemExit("partial page accepted")
    except he.HostEncryptionError: pass

    # stress: random backups and loads on disjoint page sets, checked against a model
    enc2 = he.HostPoolEncryption(8); pool2 = Pool(pages=256, page_size=16, token_bytes=2048)
    model = {}; errors = []
    def worker(seed, pages):
        rnd = random.Random(seed)
        try:
            for _ in range(300):
                sub = rnd.sample(pages, rnd.randint(1, 6))
                if rnd.random() < 0.5 or not all(p in model for p in sub):
                    t = enc2.begin_backup([T(pool2, sub)])
                    for p in sub:
                        model[p] = os.urandom(2 * 16 * 2048); pool2.write(p, model[p])
                    enc2.finish_backup(t, Ev(rnd.random() * 0.002))
                else:
                    t = enc2.begin_load([T(pool2, sub)])
                    for p in sub: assert pool2.read(p) == model[p], f"page {p} corrupt"
                    enc2.finish_load(t, Ev(rnd.random() * 0.002))
        except Exception as e: errors.append(repr(e))
    ths = [threading.Thread(target=worker, args=(i, list(range(i * 32, (i + 1) * 32)))) for i in range(8)]
    [t.start() for t in ths]; [t.join() for t in ths]
    assert not errors, errors
    wait_sealed(enc2, len(model))
    for p, data in model.items():
        t = enc2.begin_load([T(pool2, [p])]); assert pool2.read(p) == data; enc2.finish_load(t, Ev())
    wait_sealed(enc2, len(model))

    # throughput: 2 GiB of 8 MiB pages
    enc3 = he.HostPoolEncryption(8); pool3 = Pool(pages=256, page_size=16, token_bytes=262144)
    pool3.buf[:] = 7; allp = list(range(256)); gb = pool3.buf.nbytes / 1e9
    t = enc3.begin_backup([T(pool3, allp)]); s = time.perf_counter(); enc3.finish_backup(t, Ev()); wait_sealed(enc3, 256, 60)
    print(f"seal   {gb / (time.perf_counter() - s):5.1f} GB/s (8 threads, incl. polling)")
    s = time.perf_counter(); t = enc3.begin_load([T(pool3, allp)]); d = time.perf_counter() - s
    print(f"open   {gb / d:5.1f} GB/s (8 threads) -> {d * 1e3:.0f} ms for {gb:.1f} GB")
    assert bool((pool3.buf == 7).all())
    # a page of several units: 20 MiB in two buffers
    enc4 = he.HostPoolEncryption(8); pool4 = Pool(pages=6, page_size=1, token_bytes=10 << 20)
    big = {p: os.urandom(20 << 20) for p in range(5)}
    t = enc4.begin_backup([T(pool4, list(big))])
    for p, data in big.items(): pool4.write(p, data)
    enc4.finish_backup(t, Ev()); wait_sealed(enc4, 5)
    assert all(pool4.read(p) != big[p] for p in big)
    t = enc4.begin_load([T(pool4, [4, 0])]); assert pool4.read(4) == big[4] and pool4.read(0) == big[0]
    enc4.finish_load(t, Ev()); wait_sealed(enc4, 5)
    k, _ = pool4.view(2); k[(5 << 20) + 3] ^= 4      # inside the second unit
    try: enc4.begin_load([T(pool4, [2])]); raise SystemExit("tamper in unit 1 not detected")
    except he.HostEncryptionError as e: print("multi-unit tamper refused:", e)
    s = enc3.stats(); s.pop("recent_loads"); s.pop("recent_backups"); print("ALL OK", s)
main()
