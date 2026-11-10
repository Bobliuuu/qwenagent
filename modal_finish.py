"""Finish remaining qwenagent seeds on Modal. Results land in volume qwenagent-res (survives laptop crash).
modal run --detach modal_finish.py ; then: modal volume get qwenagent-res / modal_out --force"""
import modal, subprocess, os, shutil
app = modal.App("qwenagent-finish")
vol = modal.Volume.from_name("qwenagent-res", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.12").pip_install("torch", "transformers", "numpy", "accelerate")
       .add_local_dir(".", "/app", ignore=["ckpt", "logs", "results", "traces", "figures", "viz", "__pycache__", ".git"]))

@app.function(image=img, gpu=["A10G", "L4", "L40S", "A100"], timeout=4 * 3600, volumes={"/res": vol}, retries=1)
def job(script, mode, seed):
    os.makedirs("/root/venv-baseline/bin", exist_ok=True)
    py = "/root/venv-baseline/bin/python"
    if not os.path.exists(py): os.symlink(subprocess.check_output(["which", "python"]).decode().strip(), py)
    for d in ("ckpt", "logs", "results", "traces"): os.makedirs(f"/app/{d}", exist_ok=True)
    env = dict(os.environ, HOME="/root")
    subprocess.run(["bash", f"/app/{script}", mode, str(seed)], cwd="/app", env=env)
    tag = script.replace("run_", "").replace(".sh", "") + f"_{mode}_s{seed}"
    os.makedirs("/res/results", exist_ok=True); os.makedirs("/res/logs", exist_ok=True); os.makedirs("/res/traces", exist_ok=True)
    for f in os.listdir("/app/results"): shutil.copy(f"/app/results/{f}", "/res/results/")
    for f in os.listdir("/app/logs"): shutil.copy(f"/app/logs/{f}", "/res/logs/")
    for f in os.listdir("/app/traces"): shutil.copy(f"/app/traces/{f}", "/res/traces/")
    vol.commit(); return tag, os.path.exists(f"/app/results/{tag}.json")

JOBS = ([("run_q7.sh", m, s) for m in ("oracle", "uniform") for s in (2, 3, 4)]
      + [("run_q9.sh", m, s) for m in ("multi", "oracle", "uniform") for s in (2, 3, 4)]
      + [("run_q9.sh", m, s) for m in ("adapter", "single") for s in range(5)]
      + [("run_q8.sh", m, 2) for m in ("adapter", "single")]
      + [("run_q8.sh", m, s) for m in ("multi", "oracle") for s in (0, 1, 2)])

@app.local_entrypoint()
def main():
    for r in job.starmap(JOBS): print(r)
