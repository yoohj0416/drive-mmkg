class GeneralCfg:
    def __init__(self):

        self.validation_session_set = [
            '201704101118', '201704130952', '201704131020', '201704131047',
            '201704131123', '201704131537', '201704131634', '201704131655',
            '201704140944', '201704141033', '201704141055', '201704141117',
            '201704141145', '201704141243', '201704141420', '201704141608',
            '201704141639', '201704141725', '201704150933', '201704151035',
            '201704151103', '201704151140', '201704151315', '201704151347',
            '201704151502', '201706061140', '201706061309', '201706061536',
            '201706061647', '201706140912', '201710031458', '201710031645',
            '201710041102', '201710041209', '201710041351', '201710041448',
        ]

        self.dataset_root = "/path/to/dataset"
        self.target_root = self.dataset_root + "/target"
        self.cause_root = self.dataset_root + "/cause"
        self.sensor_root = self.dataset_root + "/sensor"

        self.target_int_to_str = {
            0: "background",
            1: "intersection passing",
            2: "left turn",
            3: "right turn",
            4: "left lane change",
            5: "right lane change",
            6: "left lane branch",
            7: "right lane branch",
            8: "crosswalk passing",
            9: "railroad passing",
            10: "merge",
            11: "U-turn"
        }

        self.cause_int_to_str = {
            0: "background",
            1: "congestion",
            2: "sign",
            3: "red light",
            4: "crossing vehicle",
            5: "parked vehicle",
            6: "crossing pedestrian"
        }

        self.sensor_metadata = {
            "AccelPedal": {
                "name": "Accelerator Pedal", "col": 0, "unit": "%", 
                "desc": "Accelerator pedal position"
            },
            "SteeringWheelAngle": {
                "name": "Steering Wheel Angle", "col": 2, "unit": "deg",
                "desc": "Steering wheel angle. Negative: Left, Positive: Right"
            },
            "Speed": {
                "name": "Vehicle Speed", "col": 3, "unit": "ft/s",
                "desc": "Vehicle speed in feet per second"
            },
            "BrakePedal": {
                "name": "Brake Pedal", "col": 4, "unit": "kPa",
                "desc": "Brake pedal pressure"
            },
            "TurnSignal": {
                "name": "Turn Signal", "col": 6, "unit": "boolean",
                "desc": "Turn signal status"
            }
        }

        self.sample_rate = 3  # Hz