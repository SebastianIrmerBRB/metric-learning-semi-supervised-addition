import optuna

db = "optuna_study.db"

study = optuna.load_study(
    study_name="hoffer_entropy_search",
    storage=f"sqlite:///{db}",
)

obsolete = (
    "ssl_config.method_params.regularizer_params.num_compared"
)

affected = [
    trial.number
    for trial in study.trials
    if obsolete in trial.params or obsolete in trial.distributions
]

print("Trials:", len(study.trials))
print("Best trial:", study.best_trial.number)
print("Trials still containing obsolete parameter:", affected)
