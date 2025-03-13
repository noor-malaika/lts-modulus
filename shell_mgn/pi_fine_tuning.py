

try:
    pass
except Exception:
    pass

try:
    pass
except Exception:
    raise ImportError(
        "Stokes Dataset requires the pyvista library. Install with "
        + "pip install pyvista"
    )


from modulus.sym.eq.pde import PDE
from sympy import Function, Number, Symbol



class Elasticity(PDE):
    """Linear Elasticity Equations"""

    def __init__(self, E, nu, rho, dim=3):
        """
        Parameters:
        - E: Young's modulus (float or function)
        - nu: Poisson's ratio (float or function)
        - rho: Density of the material (float or function)
        - dim: Spatial dimensions (2D or 3D)
        """
        self.dim = dim

        # Define spatial coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        input_variables = {"x": x, "y": y, "z": z}
        if self.dim == 2:
            input_variables.pop("z")

        # Define displacement components
        u = Function("u")(*input_variables)  # Displacement in x
        v = Function("v")(*input_variables)  # Displacement in y
        if self.dim == 3:
            w = Function("w")(*input_variables)  # Displacement in z
        else:
            w = Number(0)

        # Convert material properties to functions if needed
        if isinstance(E, str):
            E = Function(E)(*input_variables)
        elif isinstance(E, (float, int)):
            E = Number(E)

        if isinstance(nu, str):
            nu = Function(nu)(*input_variables)
        elif isinstance(nu, (float, int)):
            nu = Number(nu)

        if isinstance(rho, str):
            rho = Function(rho)(*input_variables)
        elif isinstance(rho, (float, int)):
            rho = Number(rho)

        # Lame parameters (λ and μ) from E and ν
        λ = (E * nu) / ((1 + nu) * (1 - 2 * nu))  # First Lame parameter
        μ = E / (2 * (1 + nu))  # Shear modulus (second Lame parameter)

        # Define strain tensor components (ε_ij)
        ε_xx = u.diff(x)
        ε_yy = v.diff(y)
        ε_zz = w.diff(z) if self.dim == 3 else Number(0)
        ε_xy = (u.diff(y) + v.diff(x)) / 2
        ε_xz = (u.diff(z) + w.diff(x)) / 2 if self.dim == 3 else Number(0)
        ε_yz = (v.diff(z) + w.diff(y)) / 2 if self.dim == 3 else Number(0)

        # Define stress tensor components using Hooke's law (σ_ij)
        σ_xx = λ * (ε_xx + ε_yy + ε_zz) + 2 * μ * ε_xx
        σ_yy = λ * (ε_xx + ε_yy + ε_zz) + 2 * μ * ε_yy
        σ_zz = λ * (ε_xx + ε_yy + ε_zz) + 2 * μ * ε_zz
        σ_xy = 2 * μ * ε_xy
        σ_xz = 2 * μ * ε_xz
        σ_yz = 2 * μ * ε_yz

        # Define the governing equations (Navier-Cauchy Equations)
        self.equations = {}
        self.equations["momentum_x"] = (
            σ_xx.diff(x) + σ_xy.diff(y) + σ_xz.diff(z) - rho * u
        )
        self.equations["momentum_y"] = (
            σ_xy.diff(x) + σ_yy.diff(y) + σ_yz.diff(z) - rho * v
        )
        if self.dim == 3:
            self.equations["momentum_z"] = (
                σ_xz.diff(x) + σ_yz.diff(y) + σ_zz.diff(z) - rho * w
            )
        else:
            self.equations.pop("momentum_z", None)
