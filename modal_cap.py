"""Track B (q11) on Modal: own app qwenagent-cap, own volume qwenagent-cap-res.  One container = one cell; seeds run in parallel
processes on the same GPU (the loop is python-overhead bound, so ~3 procs/GPU is cheaper per result).
  modal run --detach modal_cap.py --batch smoke|A|B|C ;  then: modal volume get qwenagent-cap-res / modal_cap_out --force"""
import modal, subprocess, os, shutil
app = modal.App("qwenagent-cap")
vol = modal.Volume.from_name("qwenagent-cap-res", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.12").pip_install("torch", "transformers", "numpy", "accelerate")
       .add_local_dir(".", "/app", ignore=["ckpt", "logs", "results", "traces", "figures", "viz", "__pycache__", ".git"]))

@app.function(image=img, gpu=["A10G", "L4"], cpu=4.0, memory=16384, timeout=6 * 3600, volumes={"/res": vol})
def cell(name, seeds):
    os.makedirs("/root/venv-baseline/bin", exist_ok=True)
    py = "/root/venv-baseline/bin/python"
    if not os.path.exists(py): os.symlink(subprocess.check_output(["which", "python"]).decode().strip(), py)
    for d in ("logs", "results"): os.makedirs(f"/app/{d}", exist_ok=True)
    env = dict(os.environ, HOME="/root", **({"Q11_STEPS": "200", "Q11_NEVAL": "10"} if name == "cap_multi_d4" and seeds == (9,) else {}))
    ps = [subprocess.Popen(["bash", "/app/run_q11.sh", name, str(s)], cwd="/app", env=env) for s in seeds]
    for p in ps: p.wait()
    for d in ("results", "logs"):
        os.makedirs(f"/res/{d}", exist_ok=True)
        for f in os.listdir(f"/app/{d}"):
            if f.startswith("q11_"): shutil.copy(f"/app/{d}/{f}", f"/res/{d}/")
    vol.commit()
    return name, [os.path.exists(f"/app/results/q11_{name}_s{s}.json") for s in seeds]

S3 = (0, 1, 2)
BATCHES = {
 "smoke": [("cap_multi_d4", (9,))],
 # A: decisive cells, small widths (capacity binds hardest)
 "A": [(f"cap_{v}_d4", S3) for v in ("multi", "oracle", "single_eq", "single_ds", "single_max", "adapter")]
    + [(f"cap_{v}_d16", S3) for v in ("multi", "oracle", "single_eq", "single_ds")],
 # B: ops-per-register wall + large widths + uniform
 "B": [(f"opr_{m}", S3) for m in ("single", "multi", "oracle")]
    + [(f"cap_{v}_d64", S3) for v in ("multi", "oracle", "single_eq", "single_ds")] + [("cap_uniform_d4", S3)],
 # C: d_s=8 and leftovers
 "C": [(f"cap_{v}_d8", S3) for v in ("multi", "oracle", "single_eq", "single_ds", "single_max")] + [("cap_single_max_d16", S3)],
}

@app.local_entrypoint()
def main(batch: str = "smoke"):
    for r in cell.starmap(BATCHES[batch]): print(r)
