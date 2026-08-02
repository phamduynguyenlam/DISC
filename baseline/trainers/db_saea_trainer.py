import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import trainer as base_trainer


def train_db_saea_ddqn_ray(
    problem_name="ZDT1",
    dim=30,
    seed=0,
    epoch=None,
    reward_scheme=1,
    num_workers=None,
    device=None,
):
    return base_trainer.train_disc_ddqn_ray(
        problem_name=problem_name,
        dim=dim,
        seed=seed,
        epoch=epoch,
        gamma=0.99,
        reward_scheme=reward_scheme,
        reward_norm=True,
        surrogate_model="gp",
        solver="hybrid",
        hybrid_nsga3_steps=None,
        hybrid_moead_ego_steps=None,
        num_workers=num_workers,
        eval_batch_size=1,
        updates_per_epoch=None,
        hidden_dim=None,
        device=device,
        rollout_device="cpu",
        surrogate_device="cpu",
        use_ray=False,
        agent_name="db_saea",
        cuda_cleanup_before_update=True,
        cuda_cleanup_after_update=True,
    )


def main():
    args = base_trainer.parse_args()
    train_db_saea_ddqn_ray(
        problem_name=args.problem,
        dim=int(args.dim),
        seed=int(args.seed),
        epoch=args.epoch,
        reward_scheme=int(args.reward_scheme),
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
