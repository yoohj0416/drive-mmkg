from pathlib import Path
import numpy as np
import pickle as pkl

from cfg import GeneralCfg


def main():

    cfg = GeneralCfg()
    validation_session_set = cfg.validation_session_set

    cache_path = Path(cfg.dataset_root) / "EAF_parsing/saved_index.pkl"
    cache_data = pkl.load(open(cache_path, "rb"))

    sampling_frequency = 3

    goal_oriented_dir = Path(cfg.target_root)
    target_dir = Path(cfg.cause_root)
    target_dir.mkdir(parents=True, exist_ok=True)

    # The following is used for parsing labels from "Cause" layer
    layer='Cause'

    interesting_keys = [k for k, v in cache_data['layer_ix'].items() if layer in v]

    events = cache_data['events_pd'].loc[cache_data['events_pd']["layer"].isin(interesting_keys)]

    for session_id in validation_session_set:
        print(f"Processing session: {session_id}")

        # Load session_id.npy from the goal_oriented_dir and get total_length
        goal_path = goal_oriented_dir / f"{session_id}.npy"
        goal_np = np.load(goal_path)
        total_length = goal_np.shape[0]

        target_dict = np.zeros(total_length, np.int32)
        if len(events[events['session_id'] == session_id]) != 0:
            for row in events[events['session_id'] == session_id].iterrows():
                start, end = int(row[1]["start"] / 1000 * sampling_frequency), int(row[1]["end"] / 1000 * sampling_frequency)
                event_type = row[1]["event_type"]
                if event_type > 15 and event_type < 23:
                    if event_type == 21:
                        continue
                    elif event_type == 22:
                        target_dict[start: end] = event_type - 15 - 1
                    else:
                        target_dict[start: end] = event_type - 15

            np.save(target_dir / f"{session_id}.npy", target_dict)
        else:
            np.save(target_dir / f"{session_id}.npy", target_dict)

if __name__ == "__main__":
    main()