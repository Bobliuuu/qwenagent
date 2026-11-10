"""q12 realistic-task ladder on Modal (own app + volume). modal run --detach modal_real.py --batch pilot|ledger|files
Results land in volume qwenagent-real-res; fetch: modal volume get qwenagent-real-res / modal_real_out --force"""
import modal, subprocess, os, shutil
app = modal.App("qwenagent-real")
vol = modal.Volume.from_name("qwenagent-real-res", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.12").pip_install("torch", "transformers", "numpy", "accelerate")
       .add_local_dir(".", "/app", ignore=["ckpt", "logs", "results", "traces", "figures", "viz", "__pycache__", ".git"]))

@app.function(image=img, gpu=["A10G", "L4"], timeout=4 * 3600, volumes={"/res": vol}, retries=1)
def job(task, mode, seed):
    os.makedirs("/root/venv-baseline/bin", exist_ok=True)
    py = "/root/venv-baseline/bin/python"
    if not os.path.exists(py): os.symlink(subprocess.check_output(["which", "python"]).decode().strip(), py)
    for d in ("results", "logs"): os.makedirs(f"/app/{d}", exist_ok=True)
    subprocess.run(["bash", "/app/run_q12.sh", task, mode, str(seed)], cwd="/app", env=dict(os.environ, HOME="/root"))
    tag = f"q12_{task}_{mode}_s{seed}"
    for d in ("results", "logs"):
        os.makedirs(f"/res/{d}", exist_ok=True)
        for f in os.listdir(f"/app/{d}"): shutil.copy(f"/app/{d}/{f}", f"/res/{d}/")
    vol.commit(); return tag, os.path.exists(f"/app/results/{tag}.json")

MODES = ["none", "adapter", "single", "uniform", "multi", "oracle"]
BATCHES = {"pilot": [("ledger", m, 0) for m in ("none", "adapter", "multi")],
           "ledger": [("ledger", m, s) for m in MODES for s in range(4) if not (m in ("none", "adapter", "multi") and s == 0)],
           "files": [("files", m, s) for m in MODES for s in range(2)]}

@app.local_entrypoint()
def main(batch: str = "pilot"):
    for r in job.starmap(BATCHES[batch]): print(r)
