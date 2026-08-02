import trainer as base_trainer


def train_disc_af_ddqn_ray(
    problem_name="ZDT1",
    dim=30,
    seed=0,
    epoch=None,
    gamma=0.99,
    reward_scheme=1,
    reward_norm=True,
    surrogate_model="gp",
    solver="hybrid",
    hybrid_nsga3_steps=None,
    hybrid_moead_ego_steps=None,
    num_workers=None,
    eval_batch_size=1,
    updates_per_epoch=None,
    hidden_dim=None,
    device=None,
    rollout_device="cpu",
    surrogate_device="cpu",
    cuda_cleanup_before_update=True,
    cuda_cleanup_after_update=True,
    use_ray=False,
):
    return base_trainer.train_disc_ddqn_ray(
        problem_name=problem_name,
        dim=dim,
        seed=seed,
        epoch=epoch,
        gamma=gamma,
        reward_scheme=reward_scheme,
        reward_norm=reward_norm,
        surrogate_model=surrogate_model,
        solver=solver,
        hybrid_nsga3_steps=hybrid_nsga3_steps,
        hybrid_moead_ego_steps=hybrid_moead_ego_steps,
        num_workers=num_workers,
        eval_batch_size=eval_batch_size,
        updates_per_epoch=updates_per_epoch,
        hidden_dim=hidden_dim,
        device=device,
        rollout_device=rollout_device,
        surrogate_device=surrogate_device,
        use_ray=use_ray,
        agent_name="disc_af",
        cuda_cleanup_before_update=cuda_cleanup_before_update,
        cuda_cleanup_after_update=cuda_cleanup_after_update,
    )


if __name__ == "__main__":
    args = base_trainer.parse_args()
    train_disc_af_ddqn_ray(
        problem_name=args.problem,
        dim=int(args.dim),
        seed=int(args.seed),
        epoch=args.epoch,
        gamma=args.gamma,
        reward_scheme=int(args.reward_scheme),
        reward_norm=bool(args.reward_norm),
        surrogate_model=str(args.surrogate_model),
        solver=str(args.solver),
        hybrid_nsga3_steps=args.hybrid_nsga3_steps,
        hybrid_moead_ego_steps=args.hybrid_moead_ego_steps,
        num_workers=args.num_workers,
        eval_batch_size=int(args.batch),
        updates_per_epoch=args.updates_per_epoch,
        hidden_dim=args.hidden_dim,
        device=args.device,
        rollout_device=str(args.rollout_device),
        surrogate_device=str(args.surrogate_device),
        cuda_cleanup_before_update=(
            True if args.cuda_cleanup_before_update is None else bool(args.cuda_cleanup_before_update)
        ),
        cuda_cleanup_after_update=(
            True if args.cuda_cleanup_after_update is None else bool(args.cuda_cleanup_after_update)
        ),
        use_ray=bool(args.ray),
    )
