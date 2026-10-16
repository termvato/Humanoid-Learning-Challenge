"""Action-conditioned world model on Modal: from the robot's two camera images and its next actions, predict the
images 4 control steps (0.2 s) later. Trained on the demos built from the phone takes.

Frames are compressed with the frozen Stable Diffusion VAE (128x128 image -> 4x16x16 latent per camera); a small
FiLM-conditioned conv net predicts the change in latent. It is trained unrolled on its own predictions, so long
imagined rollouts stay usable.

Evaluation on held-out episodes:
  1. prediction error at 0.2, 1 and 2 s against "nothing moves" (copy the last frame);
  2. counterfactual: a classifier trained on real frames says whether the mug is on the plate. Imagining the
     demo's own actions should end on "on plate"; the same actions with the gripper never closing should not.
  3. videos: real vs imagined, and imagined with and without closing the gripper.

Usage: .venv/bin/modal run --detach infra/modal_world.py --datasets real3_retimed_400,real3_mean_400,oracle_mean_400
Outputs: /vol/wm/<run>/ (metrics.json, videos, model.pt) in the `handsim` volume.
"""

import json
import time
from pathlib import Path

import modal

VOL = Path("/vol")
vol = modal.Volume.from_name("handsim", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1")
    .pip_install("torch==2.7.1", "diffusers==0.33.1", "transformers==4.51.3", "accelerate", "av", "pandas",
                 "pyarrow", "numpy<2.3", "imageio[ffmpeg]", "opencv-python-headless", "safetensors")
)
app = modal.App("handsim-world")

S = 4  # control steps per prediction (0.2 s at 20 Hz)
K = 4  # unrolled predictions per training sample


def encode(ds: str, out: Path) -> None:
    """Decode both camera videos, VAE-encode at 128 px, save latents with actions and episode ids."""
    import av
    import numpy as np
    import pandas as pd
    import torch
    from diffusers import AutoencoderKL

    root = VOL / "demos" / ds
    data = pd.concat(pd.read_parquet(f) for f in sorted((root / "data").rglob("*.parquet")))
    data = data.sort_values("index")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).cuda().eval()
    lat = []
    for cam in ("observation.images.image", "observation.images.image2"):
        frames = []
        for f in sorted((root / "videos" / cam).rglob("*.mp4")):
            with av.open(str(f)) as c:
                for fr in c.decode(video=0):
                    frames.append(fr.to_ndarray(format="rgb24", width=128, height=128))
        x = np.stack(frames)
        assert len(x) == len(data), (cam, len(x), len(data))
        zs = []
        with torch.no_grad():
            for i in range(0, len(x), 512):
                b = torch.from_numpy(x[i : i + 512]).cuda().permute(0, 3, 1, 2).half() / 127.5 - 1
                zs.append((vae.encode(b).latent_dist.mean * 0.18215).cpu())
        lat.append(torch.cat(zs).numpy())
    np.savez(out, z=np.stack(lat, 1).astype(np.float16), action=np.stack(data["action"].values).astype(np.float32),
             episode=data["episode_index"].values.astype(np.int32))
    print(ds, len(data), "frames encoded", flush=True)


def build(dim: int = 192):
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, c, emb):
            super().__init__()
            self.n1, self.n2 = nn.GroupNorm(8, c), nn.GroupNorm(8, c)
            self.c1, self.c2 = nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1)
            self.film = nn.Linear(emb, 2 * c)

        def forward(self, x, e):
            g, b = self.film(e)[:, :, None, None].chunk(2, 1)
            h = self.c1(nn.functional.silu(self.n1(x)))
            h = self.n2(h) * (1 + g) + b
            return x + self.c2(nn.functional.silu(h))

    class WM(nn.Module):
        """(z_prev, z_now: [B,2,4,16,16], actions [B,S,7]) -> z_next [B,2,4,16,16]."""

        def __init__(self):
            super().__init__()
            emb = 256
            self.act = nn.Sequential(nn.Linear(S * 7, emb), nn.SiLU(), nn.Linear(emb, emb))
            self.inp = nn.Conv2d(16, dim, 3, padding=1)
            self.hi = nn.ModuleList([Block(dim, emb) for _ in range(3)])
            self.down = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
            self.lo = nn.ModuleList([Block(dim, emb) for _ in range(6)])
            self.up = nn.ConvTranspose2d(dim, dim, 4, stride=2, padding=1)
            self.hi2 = nn.ModuleList([Block(dim, emb) for _ in range(3)])
            self.out = nn.Conv2d(dim, 8, 3, padding=1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, zp, zn, a):
            B = zn.shape[0]
            e = self.act(a.reshape(B, -1))
            h = self.inp(torch.cat([zp.reshape(B, 8, 16, 16), zn.reshape(B, 8, 16, 16)], 1))
            for blk in self.hi:
                h = blk(h, e)
            s = h
            h = self.down(h)
            for blk in self.lo:
                h = blk(h, e)
            h = self.up(h) + s
            for blk in self.hi2:
                h = blk(h, e)
            return zn + self.out(h).reshape(B, 2, 4, 16, 16)

    class Done(nn.Module):
        """Mug-on-plate classifier on a latent pair (both cameras)."""

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Conv2d(8, 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, 64, 3, stride=2, padding=1),
                                     nn.SiLU(), nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.SiLU(), nn.Flatten(),
                                     nn.Linear(64 * 16, 1))

        def forward(self, z):
            return self.net(z.reshape(z.shape[0], 8, 16, 16)).squeeze(1)

    return WM, Done


@app.function(image=image, gpu="H100", cpu=8.0, memory=65536, timeout=4 * 3600, volumes={"/vol": vol})
def world(datasets: str, run: str, steps: int = 20000) -> str:
    import numpy as np
    import torch
    from diffusers import AutoencoderKL

    t0 = time.time()
    out = VOL / "wm" / run
    out.mkdir(parents=True, exist_ok=True)
    cache = VOL / "wm" / "latents"
    cache.mkdir(parents=True, exist_ok=True)
    eps = []  # (name, z [T,2,4,16,16], a [T,7])
    for ds in datasets.split(","):
        f = cache / f"{ds}.npz"
        if not f.exists():
            encode(ds, f)
            vol.commit()
        d = np.load(f)
        for e in np.unique(d["episode"]):
            m = d["episode"] == e
            eps.append((f"{ds}/{e}", d["z"][m], d["action"][m]))
    test = [i for i in range(len(eps)) if i % 20 == 0]
    train = [i for i in range(len(eps)) if i % 20 != 0]
    print(len(eps), "episodes;", len(test), "held out", f"{time.time() - t0:.0f}s", flush=True)

    dev = "cuda"
    Z = [torch.from_numpy(e[1].astype(np.float32)).to(dev) for e in eps]
    A = [torch.from_numpy(e[2]).to(dev) for e in eps]
    WM, Done = build()
    wm, done = WM().to(dev), Done().to(dev)

    def batch(ids, n):
        ei = np.random.choice(ids, n)
        zp, zn, acts, tgt = [], [], [], []
        for i in ei:
            T = len(Z[i])
            t = np.random.randint(0, max(1, T - S * K))
            idx = [min(t + S * k, T - 1) for k in range(K + 1)]
            zp.append(Z[i][max(t - S, 0)])
            zn.append(Z[i][t])
            a = torch.stack([A[i][min(t + j, T - 1)] for j in range(S * K)])
            acts.append(a.reshape(K, S, 7))
            tgt.append(torch.stack([Z[i][j] for j in idx[1:]]))
        return torch.stack(zp), torch.stack(zn), torch.stack(acts), torch.stack(tgt)

    opt = torch.optim.AdamW(wm.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.05)
    log = []
    for it in range(steps):
        zp, zn, acts, tgt = batch(train, 128)
        loss = 0
        with torch.autocast("cuda", torch.bfloat16):
            for k in range(K):
                pred = wm(zp, zn, acts[:, k])
                loss = loss + ((pred.float() - tgt[:, k]) ** 2).mean()
                zp, zn = zn, pred
        opt.zero_grad()
        (loss / K).backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 500 == 0:
            log.append({"step": it, "loss": float(loss) / K})
            print(log[-1], f"{time.time() - t0:.0f}s", flush=True)
    torch.save(wm.state_dict(), out / "model.pt")

    # mug-on-plate labels from the gripper: from the final release on = on plate (episodes end ~10 steps after)
    def labels(a):
        g = a[:, 6].cpu().numpy()
        opens = np.flatnonzero((g[:-1] > 0) & (g[1:] <= 0))
        y = np.zeros(len(g), np.float32)
        if len(opens):
            y[opens[-1] + 1 :] = 1
        return torch.from_numpy(y).to(dev)

    Y = [labels(a) for a in A]
    copt = torch.optim.AdamW(done.parameters(), lr=1e-3)
    for it in range(3000):
        ei = np.random.choice(train, 256)
        ti = [np.random.randint(len(Z[i])) for i in ei]
        z = torch.stack([Z[i][t] for i, t in zip(ei, ti)])
        y = torch.stack([Y[i][t] for i, t in zip(ei, ti)])
        l = torch.nn.functional.binary_cross_entropy_with_logits(done(z), y, pos_weight=torch.tensor(8.0, device=dev))
        copt.zero_grad()
        l.backward()
        copt.step()
    with torch.no_grad():
        acc = np.mean([float(((done(Z[i]) > 0).float() == Y[i]).float().mean()) for i in test])
        acc_end = np.mean([float(done(Z[i][-1:])[0] > 0) for i in test])  # real last frames: mug on plate

    @torch.no_grad()
    def imagine(i, acts, t0=0, n=None):
        """Open-loop imagined latents from real frame t0 given the action sequence acts [T,7]."""
        T = len(acts)
        n = n or (T - t0) // S
        zp, zn, zs = Z[i][max(t0 - S, 0)][None], Z[i][t0][None], []
        for k in range(n):
            a = torch.stack([acts[min(t0 + S * k + j, T - 1)] for j in range(S)])[None]
            zp, zn = zn, wm(zp, zn, a)
            zs.append(zn[0])
        return torch.stack(zs)

    wm.eval()
    done.eval()
    # 1. error vs horizon on held-out episodes, against copying the last frame
    err = {h: {"wm": [], "copy": []} for h in (S, 5 * S, 10 * S)}
    for i in test:
        T = len(Z[i])
        for t in range(0, T - 10 * S, 8):
            zs = imagine(i, A[i], t, 10)
            for h in err:
                k = h // S - 1
                err[h]["wm"].append(float(((zs[k] - Z[i][t + h]) ** 2).mean()))
                err[h]["copy"].append(float(((Z[i][t] - Z[i][t + h]) ** 2).mean()))
    horizon = {f"{h / 20:.1f}s": {m: float(np.mean(v)) for m, v in d.items()} for h, d in err.items()}

    # 2. counterfactual: own actions vs the same actions with the gripper never closing
    cf = {"own": [], "no_grasp": [], "real_end": []}
    for i in test:
        a_no = A[i].clone()
        a_no[:, 6] = -1.0
        for key, acts in (("own", A[i]), ("no_grasp", a_no)):
            zs = imagine(i, acts)
            cf[key].append(float(torch.sigmoid(done(zs[-1:]))[0]))
        cf["real_end"].append(float(torch.sigmoid(done(Z[i][-1:]))[0]))
    counter = {k: {"mean_p_on_plate": float(np.mean(v)), "frac_on_plate": float(np.mean(np.array(v) > 0.5))}
               for k, v in cf.items()}

    # 3. videos: decode with the VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to(dev).eval()
    import cv2
    import imageio.v3 as iio

    @torch.no_grad()
    def dec(z):  # [N,2,4,16,16] -> [N,128,256,3] uint8, both cameras side by side, upright
        x = vae.decode(z.reshape(-1, 4, 16, 16).half() / 0.18215).sample
        x = ((x.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()[:, ::-1, ::-1]
        x = x.reshape(-1, 2, 128, 128, 3)
        return np.concatenate([x[:, 0], x[:, 1]], 2)

    def label(im, text):
        im = np.ascontiguousarray(im)
        cv2.putText(im, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return im

    vids = []
    for n, i in enumerate(test[:4]):
        a_no = A[i].clone()
        a_no[:, 6] = -1.0
        real = dec(Z[i][S :: S][: (len(Z[i]) - 1) // S])
        own, no = dec(imagine(i, A[i])), dec(imagine(i, a_no))
        m = min(len(real), len(own), len(no))
        fr = [np.concatenate([label(real[k], "real"), label(own[k], "imagined, same actions"),
                              label(no[k], "imagined, gripper never closes")], 0) for k in range(m)]
        name = f"rollout_{n}.mp4"
        iio.imwrite(out / name, np.stack(fr), fps=5, codec="libx264", macro_block_size=1)
        vids.append({"file": name, "episode": eps[i][0], "steps": len(Z[i])})

    res = {"run": run, "datasets": datasets, "episodes": len(eps), "held_out": len(test), "train_steps": steps,
           "loss": log, "error_by_horizon": horizon, "classifier_acc_real_heldout": float(acc), "classifier_on_plate_real_end": float(acc_end),
           "counterfactual": counter, "videos": vids, "seconds": time.time() - t0}
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    vol.commit()
    return json.dumps({k: res[k] for k in ("error_by_horizon", "classifier_acc_real_heldout", "counterfactual")})


@app.local_entrypoint()
def main(datasets: str, run: str = "wm1", steps: int = 20000) -> None:
    print(world.remote(datasets, run, steps))
