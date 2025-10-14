import numpy as np
from scipy.optimize import minimize
from openmm import unit
from typing import Tuple, List

def openmm_energy_and_gradient(
    x: np.ndarray,
    simulation,
    eval_counter: List[int]
) -> Tuple[float, np.ndarray]:
    """
    Compute OpenMM potential energy and gradient for given atomic positions.

    Args:
        x (np.ndarray): Flattened atomic positions (shape: n_atoms*3).
        simulation (Simulation): OpenMM Simulation object.
        eval_counter (List[int]): Single-element list to count evaluations.

    Returns:
        Tuple[float, np.ndarray]: Potential energy (kJ/mol) and gradient (flattened, kJ/mol/nm).
    """
    x_nm = x.reshape((-1, 3)) * unit.nanometer
    simulation.context.setPositions(x_nm)
    state = simulation.context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
    grad = -forces.reshape(-1)
    eval_counter[0] += 1
    return energy, grad

def minimize_with_counter(
    simulation,
    maxiter: int = 100000
) -> int:
    """
    Minimize energy using L-BFGS-B and count energy/gradient evaluations.

    Args:
        simulation (Simulation): OpenMM Simulation object.
        maxiter (int, optional): Maximum number of optimizer iterations. Defaults to 100000.

    Returns:
        int: Number of energy/gradient evaluations performed.
    """
    state = simulation.context.getState(getPositions=True)
    x0 = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer).reshape(-1)
    eval_counter = [0]
    result = minimize(
        fun=lambda x: openmm_energy_and_gradient(x, simulation, eval_counter),
        x0=x0,
        method="L-BFGS-B",
        jac=True,
        options={"gtol": 10.0, "maxiter": maxiter}
    )
    x_final = result.x.reshape((-1, 3)) * unit.nanometer
    simulation.context.setPositions(x_final)
    return eval_counter[0]