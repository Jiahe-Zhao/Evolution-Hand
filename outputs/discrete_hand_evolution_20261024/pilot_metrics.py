"""Observe complete training episodes without resetting the simulator."""
import json
import time


def install(agent, out):
    original = agent.train_epoch
    start = time.monotonic()
    previous = {'episodes': 0, 'successful_terminal_episodes': 0}

    def train_epoch():
        result = original()
        data = agent.vec_env.diagnostics()
        count = data['episodes'] - previous['episodes']
        success = data['successful_terminal_episodes'] - previous['successful_terminal_episodes']
        row = dict(epoch=int(agent.epoch_num), elapsed_s=time.monotonic()-start,
                   completed_episodes=data['episodes'], successes=data['successful_terminal_episodes'],
                   new_episodes=count, new_successes=success,
                   new_success_rate=success/count if count else None,
                   learning_rate=float(agent.last_lr), falls=data['falls'],
                   max_speed_m_s=data['max_speed_m_s'], max_digit_force_N=data['max_digit_force_N'],
                   anomaly_count=len(data['anomalies']))
        with (out/'learning_curve.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        temporary=out/'progress.tmp';temporary.write_text(json.dumps(row,indent=2));temporary.replace(out/'progress.json')
        previous.update({k:data[k] for k in previous})
        return result
    agent.train_epoch = train_epoch
