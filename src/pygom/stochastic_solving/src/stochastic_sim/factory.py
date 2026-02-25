"""
Factory to assemble solver components from user input
"""

from .config import SolverConfig
from .config import TauConfig, FixedTauConfig, Cao2006TauConfig
from .config import ExactConfig, DirectMethodConfig, FirstReactionMethodConfig
from .core.exact import DirectReaction, FirstReaction
from .core.tau import TauLeap
from .core.tau.method import Fixed, Cao2006

from .config import CriticalReactionConfig, NoCheckConfig, ForbiddenStateConfig
from .core.step_checker import CriticalReactionCheck, NoCheck, ForbiddenStateCheck

from .config import TauRefinerConfig, NoRefinerConfig, ProbabilisticRefinerConfig
from .core.tau.precaution import ProbabilisticTauPrecaution, NoTauPrecaution

def make_stepper(
    *,
    config: SolverConfig,
    transition_func,
    state_change_mat,
    x_min,
    x_max,
    proceed_if_rates_zero,
    transition_var_func = None,
    transition_mean_func = None
):
    """
    Construct a complete stochastic stepper
    """

    # ============================================================
    # EXACT FAMILY
    # ============================================================
    if isinstance(config, ExactConfig):
        method_cfg = config.method

        if isinstance(method_cfg, DirectMethodConfig):
            return DirectReaction(
                transition_func=transition_func,
                state_change_mat=state_change_mat,
                x_min=x_min,
                x_max=x_max,
                proceed_if_rates_zero=proceed_if_rates_zero,
            )

        elif isinstance(method_cfg, FirstReactionMethodConfig):
            return FirstReaction(
                transition_func=transition_func,
                state_change_mat=state_change_mat,
                x_min=x_min,
                x_max=x_max,
                proceed_if_rates_zero=proceed_if_rates_zero,
            )

        else:
            raise TypeError(f"Unknown Exact family method: {type(method_cfg)}")

    # ============================================================
    # TAU FAMILY
    # ============================================================
    elif isinstance(config, TauConfig):
        # ------------------------
        # Build tau method
        # ------------------------
        method_cfg = config.method

        if isinstance(method_cfg, FixedTauConfig):
            tau_method = Fixed(
                transition_func,
                state_change_mat,
                tau=method_cfg.tau,
            )

        elif isinstance(method_cfg, Cao2006TauConfig):
            tau_method = Cao2006(
                transition_func,
                state_change_mat,
                transition_mean_func=transition_mean_func,
                transition_var_func=transition_var_func,
                epsilon=method_cfg.epsilon,
            )

        # elif isinstance(method_cfg, Alternative2026TauConfig):
        #     tau_method = Alternative2026(
        #         transition_func,
        #         state_change_mat,
        #         transition_mean_func=transition_mean_func,
        #         transition_var_func=transition_var_func,
        #         epsilon=method_cfg.epsilon,
        #     )

        else:
            raise TypeError(f"Unknown Tau method family: {type(method_cfg)}")

        # ------------------------
        # Build checker
        # ------------------------
        checker_cfg = config.checker

        if isinstance(checker_cfg, CriticalReactionConfig):
            checker = CriticalReactionCheck()

        elif isinstance(checker_cfg, ForbiddenStateConfig):
            checker = ForbiddenStateCheck()

        elif isinstance(checker_cfg, NoCheckConfig):
            checker = NoCheck()

        else:
            raise TypeError(f"Unknown checker config: {type(checker_cfg)}")

        # ------------------------
        # Build refiner
        # ------------------------
        refiner_cfg = config.refiner

        if isinstance(refiner_cfg, ProbabilisticRefinerConfig):
            refiner = ProbabilisticTauPrecaution(
                max_retries=refiner_cfg.max_retries,
                acceptable_prob_misstep=refiner_cfg.acceptable_prob_misstep,
                factor_min=refiner_cfg.factor_min,
                factor_max=refiner_cfg.factor_max,
            )

        elif isinstance(refiner_cfg, NoRefinerConfig):
            refiner = NoTauPrecaution()

        else:
            raise TypeError(f"Unknown refiner config: {type(refiner_cfg)}")

        # ------------------------
        # Construct TauLeap
        # ------------------------
        return TauLeap(
            transition_func=transition_func,
            state_change_mat=state_change_mat,
            x_min=x_min,
            x_max=x_max,
            proceed_if_rates_zero=proceed_if_rates_zero,
            retry_max=config.retry_max,
            tau_rescale=config.tau_rescale,
            tau_method=tau_method,
            proposal_checker=checker,
            tau_refiner=refiner,
        )

    # ============================================================
    # Unknown config
    # ============================================================
    else:
        raise TypeError(f"Unknown solver config type: {type(config)}")
