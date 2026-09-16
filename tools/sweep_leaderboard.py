#!/usr/bin/env python3
import json
import glob
import os
import sys
import datetime

def main():
    task_filter = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    files = glob.glob('logs/drone/*.json')
    if not files:
        print("No log files found in logs/drone/")
        return

    runs = []
    for f in files:
        try:
            with open(f) as fp:
                d = json.load(fp)
            if d.get('env', {}).get('task') == task_filter:
                score = d.get('metrics', {}).get('env/score', [-1])[-1]
                perf = d.get('metrics', {}).get('env/perf', [0])[-1]
                rings = d.get('metrics', {}).get('env/rings_passed', [0])[-1]
                cols = d.get('metrics', {}).get('env/ring_collisions', [0])[-1]
                ep_ret = d.get('metrics', {}).get('env/episode_return', [0])[-1]
                env = d.get('env', {})
                train = d.get('train', {})
                policy = d.get('policy', {})
                mtime = os.path.getmtime(f)
                runs.append({
                    'id': os.path.basename(f).replace('.json', ''),
                    'mtime': mtime,
                    'score': score if score is not None else -1,
                    'perf': perf if perf is not None else 0,
                    'rings': rings if rings is not None else 0,
                    'cols': cols if cols is not None else 0,
                    'ep_ret': ep_ret if ep_ret is not None else 0,
                    'lr': train.get('learning_rate', 0),
                    'gamma': train.get('gamma', 0),
                    'horizon': train.get('horizon', 64),
                    'hidden': policy.get('hidden_size', 64),
                    'alpha_dist': env.get('alpha_dist', 0),
                    'ring_reward': env.get('ring_reward', 0),
                    'col_pen': env.get('collision_penalty', 0),
                    'env': env,
                    'train': train,
                    'policy': policy,
                })
        except Exception:
            continue

    if not runs:
        print(f"No completed runs found matching task {task_filter}.")
        return

    # Sort by score descending
    runs.sort(key=lambda x: x['score'], reverse=True)

    header = f"{'Rank':<5} {'Score':<8} {'Return':<9} {'Rings':<7} {'Cols':<7} {'Horizon':<8} {'Hidden':<7} {'LR':<9} {'Gamma':<7} {'AlphaDist':<10} {'RingRew':<8} {'ColPen':<7} {'Run ID'}"
    sep = '-' * len(header)

    print(f"\n{'='*len(header)}")
    print(f"  PUFFERLIB SWEEP LEADERBOARD (Task {task_filter} - Total Runs: {len(runs)})")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for idx, r in enumerate(runs[:15], 1):
        print(f"#{idx:<4} {r['score']:<8.4f} {r['ep_ret']:<9.1f} {r['rings']:<7.4f} {r['cols']:<7.4f} {r['horizon']:<8} {r['hidden']:<7} {r['lr']:<9.5f} {r['gamma']:<7.4f} {r['alpha_dist']:<10.3f} {r['ring_reward']:<8.3f} {r['col_pen']:<7.3f} {r['id']}")

    # Also display the 5 most recent runs so user can verify real-time progress
    recent_runs = sorted(runs, key=lambda x: x['mtime'], reverse=True)[:5]
    print(f"\n{'-'*len(header)}")
    print(f"  RECENTLY COMPLETED RUNS (Latest 5)")
    print(f"{'-'*len(header)}")
    print(f"{'Finished':<9} {'Score':<8} {'Return':<9} {'Rings':<7} {'Cols':<7} {'Horizon':<8} {'Hidden':<7} {'LR':<9} {'Gamma':<7} {'AlphaDist':<10} {'RingRew':<8} {'ColPen':<7} {'Run ID'}")
    print(sep)
    for r in recent_runs:
        t_str = datetime.datetime.fromtimestamp(r['mtime']).strftime('%H:%M:%S')
        print(f"{t_str:<9} {r['score']:<8.4f} {r['ep_ret']:<9.1f} {r['rings']:<7.4f} {r['cols']:<7.4f} {r['horizon']:<8} {r['hidden']:<7} {r['lr']:<9.5f} {r['gamma']:<7.4f} {r['alpha_dist']:<10.3f} {r['ring_reward']:<8.3f} {r['col_pen']:<7.3f} {r['id']}")

    best = runs[0]
    print(f"\n{'='*len(header)}")
    print(f"🏆 Current Best Run: {best['id']} (Score: {best['score']:.4f}, Return: {best['ep_ret']:.1f}, Rings: {best['rings']:.4f})")
    print(f"{'-'*len(header)}")
    print("Complete Configuration for config/drone.ini:")
    print(f"{'-'*len(header)}")

    print("[env]")
    print(f"alpha_dist = {best['env'].get('alpha_dist', 1.0):.6f}")
    print(f"ring_reward = {best['env'].get('ring_reward', 1.0):.6f}")
    print(f"collision_penalty = {best['env'].get('collision_penalty', 0.5):.6f}")

    print("\n[policy]")
    for k, v in sorted(best['policy'].items()):
        print(f"{k} = {v}")

    print("\n[train]")
    # Print sorted training parameters
    skip_train = {'gpus', 'seed'}
    for k, v in sorted(best['train'].items()):
        if k in skip_train:
            continue
        if isinstance(v, float):
            print(f"{k} = {v:.6g}")
        else:
            print(f"{k} = {v}")

    print(f"{'='*len(header)}\n")

if __name__ == '__main__':
    main()
