from prefect import flow, task
from subprocess import run, CalledProcessError
import os

@task
def train(cfg="configs/prod.yaml"):
    cmd = ["python","pipelines/train.py",]
    env = os.environ.copy()
    env["QF_CONFIG"]=cfg
    res = run(cmd, check=True, env=env)
    return res.returncode

@flow(name="quantumflow-nightly")
def nightly_flow():
    train()

if __name__ == "__main__":
    nightly_flow()
