"""Track D robustness sweep (layers, model scale) on Modal. Own app + volume; results survive laptop crash.
  modal run --detach modal_scale.py --batch layers     # 0.6B layers 4,8,18,24 x adapter,single,multi x seeds 0-2
  modal run --detach modal_scale.py --batch s17        # Qwen3-1.7B layer 12
  modal run --detach modal_scale.py --batch s4         # Qwen3-4B layer 15
then: modal volume get qwenagent-scale-res / modal_scale_out --force"""
import modal, subprocess, os, shutil
app = modal.App("qwenagent-scale")
vol = modal.Volume.from_name("qwenagent-scale-res", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.12").pip_install("torch", "transformers", "numpy", "accelerate")
       .add_local_dir(".", "/app", ignore=["ckpt", "logs", "results", "traces", "figures", "viz", "__pycache__", ".git"]))

def _run(mode, seed, model, layer):
    os.makedirs("/root/venv-baseline/bin", exist_ok=True)
    py = "/root/venv-baseline/bin/python"
    if not os.path.exists(py): os.symlink(subprocess.check_output(["which", "python"]).decode().strip(), py)
    for d in ("logs", "results"): os.makedirs(f"/app/{d}", exist_ok=True)
    env = dict(os.environ, HOME="/root")
    subprocess.run(["bash", "/app/run_q13.sh", mode, str(seed), model, str(layer)], cwd="/app", env=env)
    tag = f"q13_{model.split('/')[-1].replace('Qwen3-', '')}_L{layer}_{mode}_s{seed}"
    for d in ("results", "logs"):
        os.makedirs(f"/res/{d}", exist_ok=True)
        for f in os.listdir(f"/app/{d}"):
            if f.startswith("q13_"): shutil.copy(f"/app/{d}/{f}", f"/res/{d}/")
    vol.commit(); return tag, os.path.exists(f"/app/results/{tag}.json")

@app.function(image=img, gpu=["A10G", "L4"], timeout=6 * 3600, volumes={"/res": vol}, retries=1)
def job_small(mode, seed, model, layer): return _run(mode, seed, model, layer)

@app.function(image=img, gpu=["L40S", "A100"], timeout=8 * 3600, volumes={"/res": vol}, retries=1)
def job_big(mode, seed, model, layer): return _run(mode, seed, model, layer)

B06, B17, B4 = "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"
BATCHES = {
  "layers": ("small", [(m, s, B06, l) for l in (4, 8, 18, 24) for m in ("adapter", "single", "multi") for s in range(3)]),
  "s17": ("small", [(m, s, B17, 12) for m in ("none", "adapter", "single", "multi") for s in range(3)]),
  "s4": ("big", [(m, s, B4, 15) for m in ("none", "adapter", "single", "multi") for s in range(2)]),
  # budget-trimmed 4B set (1 seed none/multi, 2 seeds adapter/single)
  "s4lite": ("big", [("none", 0, B4, 15), ("multi", 0, B4, 15)] + [(m, s, B4, 15) for m in ("adapter", "single") for s in range(2)]),
}

@app.local_entrypoint()
def main(batch: str = "layers"):
    kind, jobs = BATCHES[batch]
    fn = job_small if kind == "small" else job_big
    for r in fn.starmap(jobs): print(r)
