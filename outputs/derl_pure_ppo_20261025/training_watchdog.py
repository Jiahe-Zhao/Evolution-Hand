"""Bounded wait for this launcher's own worker; log churn is not progress."""
import json,time

def wait_for_progress(process,out,timeout_s=600):
    started=time.time()
    while process.poll() is None:
        markers=[out/'simulation_heartbeat.json',out/'progress.json',out/'completed.json']
        last=max([started]+[p.stat().st_mtime for p in markers if p.exists()])
        if time.time()-last>timeout_s:
            (out/'watchdog_failure.json').write_text(json.dumps(dict(
                reason='No simulation or optimizer progress',timeout_s=timeout_s,pid=process.pid),indent=2))
            process.terminate()
            try:process.wait(timeout=5)
            except __import__('subprocess').TimeoutExpired:
                process.kill();process.wait()
            return 124
        time.sleep(5)
    return process.returncode
