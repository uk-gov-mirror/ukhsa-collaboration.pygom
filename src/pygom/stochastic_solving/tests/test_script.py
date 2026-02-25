# import pandas as pd
# import argparse
# import os

# import numpy as np
# from pygom import Event, Transition, SimulateOde
# from pygom.stochastic_solving.src.stochastic_sim.api import solve
# from pygom.stochastic_solving.src.stochastic_sim.config import TauConfig, FixedTauConfig, Cao2006TauConfig

# from pygom.stochastic_solving.src.stochastic_sim.config import ExactConfig, DirectMethodConfig, FirstReactionMethodConfig

# #################
# # Parse arguments
# #################

# parser = argparse.ArgumentParser()
# parser.add_argument("--method")
# parser.add_argument("--tau", type=float)
# parser.add_argument("--epsilon", type=float)
# parser.add_argument("--N", type=int)
# parser.add_argument("--R0", type=float)
# parser.add_argument("--trec", type=float)
# # parser.add_argument("--iter", type=int)
# parser.add_argument("--outdir")
# args = parser.parse_args()

# N = args.N
# R0 = args.R0
# gamma = 1/args.trec
# beta = R0 * gamma

# method = args.method

# ##############
# # Set up model
# ##############

# # Initial conditions
# i0 = 1
# x0 = np.array([N-i0, i0, 0])

# # Params and states
# params=['beta', 'gamma', 'N']
# states=['S', 'I', 'R']

# # State Limits
# xlims = np.asarray([(0, N), (0, N), (0, N)], dtype=object)
# mins = np.array([lim[0] if lim[0] is not None else -np.inf for lim in xlims])
# maxs = np.array([lim[1] if lim[1] is not None else np.inf for lim in xlims])

# # Transitions
# transition_infection=Transition(origin='S', destination='I', transition_type='T', magnitude='1')
# event_infection=Event(transition_list=[transition_infection], rate='beta*S*I/N')
# transition_recovery=Transition(origin='I', destination='R', transition_type='T', magnitude='1')
# event_recovery=Event(transition_list=[transition_recovery], rate='gamma*I')

# model=SimulateOde(state=states, param=params, event=[event_infection, event_recovery])
# model.parameters = {"beta": beta, "gamma": gamma, "N":N}

# # Functions to input to solvers
# transition_func = model.event_rate_vector
# state_change_mat = model.state_change_matrix
# transition_mean_func = model.transition_mean
# transition_var_func = model.transition_variance


# parameters = {}

# if method == "tau_fixed":
#     algo =  TauConfig(method=FixedTauConfig(tau=args.tau))
#     parameters = {"tau" : args.tau}
# if method == "tau_adaptive":
#     algo =  TauConfig(method=Cao2006TauConfig(epsilon=args.epsilon))
#     parameters = {"epsilon" : args.epsilon}
# if method == "direct":
#     algo =  ExactConfig(method = DirectMethodConfig())
# if method == "first":
#     algo =  ExactConfig(method = FirstReactionMethodConfig())


# n_iter = 100  # just an example
# rows = []
# for _ in range(n_iter):

#     ################
#     # Run simulation
#     ################

#     sim = solve(
#         x0,
#         10000,
#         transition_func,
#         state_change_mat,
#         mins, maxs,
#         config=algo,
#         perf=True,
#         proceed_if_rates_zero=False,
#         transition_mean_func = transition_mean_func,
#         transition_var_func = transition_var_func)

#     ################
#     # Process output
#     ################

#     final_size = sim.out.x[-1,2]

#     row = {
#         "method": args.method,
#         # "iter": args.iter,
#         "final_size": final_size,
#         "cpu_time_seconds": sim.performance.cpu_time_seconds,
#         "wall_time_seconds": sim.performance.wall_time_seconds,
#     }

#     row = row | parameters

#     rows.append(row)

# df = pd.DataFrame(rows)

# outfile = os.path.join(args.outdir, f"{args.method}.csv")
# df.to_csv(outfile, index=False)


import os
import argparse
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

from pygom import Event, Transition, SimulateOde
from pygom.stochastic_solving.src.stochastic_sim.api import solve
from pygom.stochastic_solving.src.stochastic_sim.config import (
    TauConfig, FixedTauConfig, Cao2006TauConfig,
    ExactConfig, DirectMethodConfig, FirstReactionMethodConfig
)

#################
# Parse arguments
#################
parser = argparse.ArgumentParser()
parser.add_argument("--method", required=True)
parser.add_argument("--tau", type=float)
parser.add_argument("--epsilon", type=float)
parser.add_argument("--N", type=int, required=True)
parser.add_argument("--R0", type=float, required=True)
parser.add_argument("--trec", type=float, required=True)
parser.add_argument("--nsim", type=int, required=True)
parser.add_argument("--outdir", required=True)
# parser.add_argument("--chunk_id", type=int, default=0)
args = parser.parse_args()

##############
# Model setup
##############
i0 = 1
x0 = np.array([args.N - i0, i0, 0])
gamma = 1 / args.trec
beta = args.R0 * gamma

params = ['beta', 'gamma', 'N']
states = ['S', 'I', 'R']
xlims = np.array([(0, args.N), (0, args.N), (0, args.N)], dtype=object)
mins = np.array([lim[0] if lim[0] is not None else -np.inf for lim in xlims])
maxs = np.array([lim[1] if lim[1] is not None else np.inf for lim in xlims])

# Transitions
t_inf = Transition(origin='S', destination='I', transition_type='T', magnitude='1')
e_inf = Event([t_inf], rate='beta*S*I/N')
t_rec = Transition(origin='I', destination='R', transition_type='T', magnitude='1')
e_rec = Event([t_rec], rate='gamma*I')

model = SimulateOde(state=states, param=params, event=[e_inf, e_rec])
model.parameters = {"beta": beta, "gamma": gamma, "N": args.N}

transition_func = model.event_rate_vector
state_change_mat = model.state_change_matrix
transition_mean_func = model.transition_mean
transition_var_func = model.transition_variance

# Choose solver
parameters = {}
if args.method == "tau_fixed":
    algo = TauConfig(method=FixedTauConfig(tau=args.tau))
    parameters = {"tau": args.tau}
elif args.method == "tau_adaptive":
    algo = TauConfig(method=Cao2006TauConfig(epsilon=args.epsilon))
    parameters = {"epsilon": args.epsilon}
elif args.method == "direct":
    algo = ExactConfig(method=DirectMethodConfig())
elif args.method == "first":
    algo = ExactConfig(method=FirstReactionMethodConfig())
else:
    raise ValueError(f"Unknown method {args.method}")

##########################
# Determine number of cores
##########################

# n_cores = int(os.environ.get("SLURM_CPUS_ON_NODE"))
n_cores = cpu_count()

print(f"Running {args.nsim} simulations on {n_cores} cores")

##########################
# Function to run one simulation
##########################
def run_one_sim(_):
    sim = solve(
        x0,
        10000,
        transition_func,
        state_change_mat,
        mins, maxs,
        config=algo,
        perf=True,
        proceed_if_rates_zero=False,
        transition_mean_func=transition_mean_func,
        transition_var_func=transition_var_func
    )
    return {
        "method": args.method,
        "final_size": sim.out.x[-1, 2],
        "cpu_time_seconds": sim.performance.cpu_time_seconds,
        "wall_time_seconds": sim.performance.wall_time_seconds,
        # "N": args.N,
        # "R0": args.R0,
        **parameters
    }

##########################
# Run simulations in parallel
##########################

def float_to_str(x, precision=6):
    s = f"{x:.{precision}f}".rstrip('0').rstrip('.')
    return s.replace('.', 'p')

def params_to_filename(params):
    parts = []
    for key in params:
        val = params[key]
        if isinstance(val, float):
            val = float_to_str(val)
        parts.append(f"{key}_{val}")
    return "__".join(parts)

epi_params = {"R0": args.R0, "trec": args.trec, "N": args.N}

fname = params_to_filename( parameters | epi_params )

if __name__ == "__main__":
    with Pool(n_cores) as pool:
        results = pool.map(run_one_sim, range(args.nsim))

    df = pd.DataFrame(results)

    ##########################
    # Save one output file per chunk
    ##########################
    os.makedirs(args.outdir, exist_ok=True)
    # outfile = os.path.join(args.outdir, f"{args.method}_chunk{args.chunk_id}.parquet")
    # df.to_parquet(outfile, index=False)
    outfile = os.path.join(args.outdir, f"{args.method}__{fname}.csv")
    df.to_csv(outfile, index=False)
    print(f"Saved {args.nsim} simulations to {outfile}")