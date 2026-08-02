import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tester as base_tester


if __name__ == "__main__":
    base_tester.main(agent_name="disc_single_dqn")
