"""
Build a driving-scene knowledge graph in Memgraph.

This script reads:
- data/description_sensor/*.json: Driving scene data with sensor information
- data/textual_kg/*.json: Action/Cause annotations
- data/id_mapping.json: Video filename to trip ID mapping

And creates a knowledge graph in Memgraph with:
- Trip nodes (one per video clip)
- Scene nodes (segmented by timeline)
- Action nodes (driving actions)
- Cause nodes (reasons for actions)
- Sensor nodes (speed, gyro, pedals, turn signal)
"""

import argparse
import glob
import json
import os
from typing import Optional

from neo4j import Driver, GraphDatabase, Session
from tqdm import tqdm

# --- Environment and Database Configuration (Memgraph) ---
URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
AUTH = (os.getenv("MEMGRAPH_USERNAME", ""), os.getenv("MEMGRAPH_PASSWORD", ""))

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_DIR = os.path.join(REPO_ROOT, "data")


def time_str_to_seconds(time_str: str) -> int:
    """Convert 'MM:SS' format to seconds."""
    parts = time_str.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return 0


def load_json_file(file_path: str) -> Optional[dict]:
    """Load a JSON file if it exists."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_id_mapping(id_mapping_path: str) -> dict:
    """Load the filename-to-ID mapping."""
    data = load_json_file(id_mapping_path)
    if not data:
        return {}
    return data.get("filename_to_id", {})


def load_sensor_data(sensor_dir: str, video_name: str) -> Optional[dict]:
    """Load sensor data for a video."""
    file_path = os.path.join(sensor_dir, f"{video_name}.json")
    return load_json_file(file_path)


def load_kg_data(kg_dir: str, video_name: str) -> Optional[dict]:
    """Load KG annotation for a video."""
    file_path = os.path.join(kg_dir, f"{video_name}.json")
    return load_json_file(file_path)


def create_trip_node(session: Session, trip_id: str, video_name: str):
    """Create a Trip node in the graph."""
    query = "CREATE (t:Trip {id: $trip_id, videoName: $video_name})"
    session.run(query, trip_id=trip_id, video_name=video_name)


def create_scene_node(
    session: Session,
    scene_id: str,
    scene_number: int,
    start_time: int,
    end_time: int,
    trip_id: str,
):
    """Create a Scene node and link it to a Trip."""
    query = (
        "CREATE (s:Scene {sceneId: $scene_id, sceneNumber: $scene_number, "
        "startTimeInSec: $start_time, endTimeInSec: $end_time})"
    )
    session.run(
        query,
        scene_id=scene_id,
        scene_number=scene_number,
        start_time=start_time,
        end_time=end_time,
    )

    session.run(
        "MATCH (t:Trip {id: $trip_id}), (s:Scene {sceneId: $scene_id}) "
        "CREATE (t)-[:HAS_SCENE]->(s)",
        trip_id=trip_id,
        scene_id=scene_id,
    )


def link_to_previous_scene(session: Session, current_scene_id: str, prev_scene_id: str):
    """Create directional relationships between adjacent scenes."""
    query = (
        "MATCH (s1:Scene {sceneId: $prev_scene_id}), (s2:Scene {sceneId: $current_scene_id}) "
        "CREATE (s1)-[:NEXT_SCENE]->(s2), (s2)-[:PREVIOUS_SCENE]->(s1)"
    )
    session.run(query, prev_scene_id=prev_scene_id, current_scene_id=current_scene_id)


def create_action_node(session: Session, scene_id: str, action_id: str, action_type: str):
    """Create an Action node and link it to a Scene."""
    query = (
        "MATCH (s:Scene {sceneId: $scene_id}) "
        "CREATE (a:Action {actionId: $action_id, actionType: $action_type}) "
        "CREATE (s)-[:HAS_ACTION]->(a)"
    )
    session.run(query, scene_id=scene_id, action_id=action_id, action_type=action_type)


def create_cause_node(
    session: Session,
    scene_id: str,
    cause_id: str,
    cause_type: str,
    action_id: str,
):
    """Create a Cause node and link it to the corresponding Action and Scene."""
    session.run(
        "CREATE (c:Cause {causeId: $cause_id, causeType: $cause_type})",
        cause_id=cause_id,
        cause_type=cause_type,
    )
    session.run(
        "MATCH (a:Action {actionId: $action_id}), (c:Cause {causeId: $cause_id}) "
        "CREATE (a)-[:CAUSED_BY]->(c)",
        action_id=action_id,
        cause_id=cause_id,
    )
    session.run(
        "MATCH (s:Scene {sceneId: $scene_id}), (c:Cause {causeId: $cause_id}) "
        "CREATE (s)-[:HAS_CAUSE]->(c)",
        scene_id=scene_id,
        cause_id=cause_id,
    )


def create_sensor_node(session: Session, scene_id: str, sensor_data: dict):
    """Create a Sensor node and link it to a Scene."""
    query = (
        "MATCH (s:Scene {sceneId: $scene_id}) "
        "CREATE (sensor:Sensor $data) "
        "CREATE (s)-[:HAS_SENSOR]->(sensor)"
    )
    session.run(query, scene_id=scene_id, data=sensor_data)


def clear_trip_graph(session: Session, trip_id: str):
    """Delete existing graph data for one trip before re-inserting."""
    session.run(
        "MATCH (t:Trip {id: $trip_id})-[:HAS_SCENE]->(s:Scene)-[:HAS_SENSOR]->(sensor:Sensor) "
        "DETACH DELETE sensor",
        trip_id=trip_id,
    )
    session.run(
        "MATCH (t:Trip {id: $trip_id})-[:HAS_SCENE]->(s:Scene)-[:HAS_CAUSE]->(c:Cause) "
        "DETACH DELETE c",
        trip_id=trip_id,
    )
    session.run(
        "MATCH (t:Trip {id: $trip_id})-[:HAS_SCENE]->(s:Scene)-[:HAS_ACTION]->(a:Action) "
        "DETACH DELETE a",
        trip_id=trip_id,
    )
    session.run(
        "MATCH (t:Trip {id: $trip_id})-[:HAS_SCENE]->(s:Scene) DETACH DELETE s",
        trip_id=trip_id,
    )
    session.run("MATCH (t:Trip {id: $trip_id}) DETACH DELETE t", trip_id=trip_id)


def process_video(driver: Driver, video_name: str, trip_id: str, sensor_data: dict, kg_data: dict):
    """Process one video and write its KG to Memgraph."""
    scenes_kg = kg_data.get("scenes", [])
    sensor_info = sensor_data.get("sensor", {})

    if not scenes_kg:
        print(f"  No scenes found in KG data for {video_name}")
        return

    speeds = sensor_info.get("speeds", [])
    gyro_yaw = sensor_info.get("gyro_yaw", [])
    accel_pedal = sensor_info.get("accel_pedal", [])
    brake_pedal = sensor_info.get("brake_pedal", [])
    turn_signal = sensor_info.get("turn_signal", [])

    speed_unit = sensor_info.get("speed_unit", "m/s")
    speed_hz = sensor_info.get("speed_hz", 1.0)
    gyro_unit = sensor_info.get("gyro_unit", "rad/s")
    gyro_hz = sensor_info.get("gyro_hz", 1.0)
    gyro_axis = sensor_info.get("gyro_axis", "x")
    accel_pedal_unit = sensor_info.get("accel_pedal_unit", "%")
    brake_pedal_unit = sensor_info.get("brake_pedal_unit", "kPa")
    turn_signal_desc = sensor_info.get("turn_signal_desc", "0=off, 1=left, 2=right")

    if gyro_axis == "x":
        gyro_desc = (
            "Yaw rate (rotation around vertical axis). "
            "Negative values indicate right turn, positive values indicate left turn."
        )
    else:
        gyro_desc = f"Gyro measurement on {gyro_axis} axis."

    with driver.session() as session:
        clear_trip_graph(session, trip_id)
        create_trip_node(session, trip_id, video_name)

        action_counter = 0
        cause_counter = 0

        for index, scene in enumerate(scenes_kg):
            scene_id = f"{trip_id}_SCENE_{index}"

            start_time = time_str_to_seconds(scene.get("start", "00:00"))
            end_time = time_str_to_seconds(scene.get("end", "00:00"))

            create_scene_node(session, scene_id, index, start_time, end_time, trip_id)

            if index > 0:
                prev_scene_id = f"{trip_id}_SCENE_{index - 1}"
                link_to_previous_scene(session, scene_id, prev_scene_id)

            action_type = scene.get("action", "")
            if action_type:
                action_id = f"{trip_id}_ACTION_{action_counter}"
                create_action_node(session, scene_id, action_id, action_type)
                action_counter += 1

                cause_type = scene.get("cause", "")
                if cause_type:
                    cause_id = f"{trip_id}_CAUSE_{cause_counter}"
                    create_cause_node(session, scene_id, cause_id, cause_type, action_id)
                    cause_counter += 1

            start_idx = start_time
            end_idx = end_time + 1

            if speeds:
                speed_values = speeds[start_idx:end_idx] if end_idx <= len(speeds) else speeds[start_idx:]
                create_sensor_node(
                    session,
                    scene_id,
                    {
                        "id": f"{scene_id}_SPEED",
                        "name": "Speed",
                        "unit": speed_unit,
                        "sampleRateHz": speed_hz,
                        "description": "Vehicle speed",
                        "values": speed_values,
                    },
                )

            if gyro_yaw:
                gyro_values = gyro_yaw[start_idx:end_idx] if end_idx <= len(gyro_yaw) else gyro_yaw[start_idx:]
                create_sensor_node(
                    session,
                    scene_id,
                    {
                        "id": f"{scene_id}_GYRO",
                        "name": "Gyro Yaw",
                        "unit": gyro_unit,
                        "sampleRateHz": gyro_hz,
                        "description": gyro_desc,
                        "values": gyro_values,
                    },
                )

            if accel_pedal:
                accel_values = (
                    accel_pedal[start_idx:end_idx]
                    if end_idx <= len(accel_pedal)
                    else accel_pedal[start_idx:]
                )
                create_sensor_node(
                    session,
                    scene_id,
                    {
                        "id": f"{scene_id}_ACCEL_PEDAL",
                        "name": "Accelerator Pedal",
                        "unit": accel_pedal_unit,
                        "sampleRateHz": 1.0,
                        "description": "Accelerator pedal position percentage",
                        "values": accel_values,
                    },
                )

            if brake_pedal:
                brake_values = (
                    brake_pedal[start_idx:end_idx]
                    if end_idx <= len(brake_pedal)
                    else brake_pedal[start_idx:]
                )
                create_sensor_node(
                    session,
                    scene_id,
                    {
                        "id": f"{scene_id}_BRAKE_PEDAL",
                        "name": "Brake Pedal",
                        "unit": brake_pedal_unit,
                        "sampleRateHz": 1.0,
                        "description": "Brake pedal pressure",
                        "values": brake_values,
                    },
                )

            if turn_signal:
                signal_values = (
                    turn_signal[start_idx:end_idx]
                    if end_idx <= len(turn_signal)
                    else turn_signal[start_idx:]
                )
                create_sensor_node(
                    session,
                    scene_id,
                    {
                        "id": f"{scene_id}_TURN_SIGNAL",
                        "name": "Turn Signal",
                        "unit": "state",
                        "sampleRateHz": 1.0,
                        "description": turn_signal_desc,
                        "values": signal_values,
                    },
                )


def main():
    """Process all videos and create the Memgraph knowledge graph."""
    parser = argparse.ArgumentParser(description="Build Memgraph KG from BDD-X data")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=DEFAULT_BASE_DIR,
        help=f"Base data directory containing description_sensor/textual_kg/id_mapping.json (default: {DEFAULT_BASE_DIR})",
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    sensor_dir = os.path.join(base_dir, "description_sensor")
    kg_dir = os.path.join(base_dir, "textual_kg")
    id_mapping_path = os.path.join(base_dir, "id_mapping.json")

    missing_paths = [
        path
        for path in [sensor_dir, kg_dir, id_mapping_path]
        if not os.path.exists(path)
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Missing required data path(s):\n- "
            + "\n- ".join(missing_paths)
            + "\nPlease download/extract the dataset under data/ or pass --base-dir."
        )

    print("Loading ID mapping...")
    filename_to_id = load_id_mapping(id_mapping_path)
    print(f"Loaded {len(filename_to_id)} video mappings")

    kg_files = sorted(glob.glob(os.path.join(kg_dir, "*.json")))
    print(f"Found {len(kg_files)} KG files")

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        print("Successfully connected to Memgraph.")
        print("-" * 60)

        processed = 0
        skipped = 0

        for kg_file in tqdm(kg_files, desc="Processing videos"):
            video_name = os.path.basename(kg_file).replace(".json", "")

            trip_id = filename_to_id.get(video_name)
            if not trip_id:
                print(f"  Warning: No ID mapping for {video_name}, skipping")
                skipped += 1
                continue

            sensor_data = load_sensor_data(sensor_dir, video_name)
            if not sensor_data:
                print(f"  Warning: No sensor data for {video_name}, skipping")
                skipped += 1
                continue

            kg_data = load_kg_data(kg_dir, video_name)
            if not kg_data:
                print(f"  Warning: No KG data for {video_name}, skipping")
                skipped += 1
                continue

            process_video(driver, video_name, trip_id, sensor_data, kg_data)
            processed += 1

        print("\n" + "=" * 60)
        print("Knowledge graph creation complete!")
        print(f"  Processed: {processed}")
        print(f"  Skipped: {skipped}")
        print(f"  Total: {len(kg_files)}")


if __name__ == "__main__":
    main()
