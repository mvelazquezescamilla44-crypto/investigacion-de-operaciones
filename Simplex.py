# simplex.py
# Método Simplex (forma estándar: Max c^T x, Ax <= b, x >= 0, b >= 0)
# Problema cargado: Max z = 3x1 + 5x2
# s.a. 2x1 + x2 <= 8 ; x1 + 2x2 <= 8 ; x1,x2 >= 0

from typing import List, Optional
import math

class SimplexResult:
    def __init__(self, status: str, z_opt: Optional[float], x_opt: Optional[List[float]],
                 iterations: int, message: str):
        self.status = status            # 'optimal', 'unbounded', 'infeasible'
        self.z_opt = z_opt
        self.x_opt = x_opt
        self.iterations = iterations
        self.message = message

def argmin_with_ties(values: List[float], eligible: List[bool]) -> Optional[int]:
    """Devuelve el índice del mínimo entre los elegibles. Si no hay elegibles, None."""
    best_idx = None
    best_val = math.inf
    for i, (v, ok) in enumerate(zip(values, eligible)):
        if ok and v < best_val - 1e-12:
            best_val = v
            best_idx = i
    return best_idx

def approx_zero(x: float, eps: float = 1e-9) -> bool:
    return abs(x) <= eps

def simplex_max(A: List[List[float]], b: List[float], c: List[float],
                show_steps: bool = False) -> SimplexResult:
    """
    Resuelve: max c^T x, s.a. A x <= b, x >= 0, b >= 0
    Devuelve SimplexResult con z_opt y x_opt (solo variables originales).
    """
    if not A or not A[0]:
        return SimplexResult("infeasible", None, None, 0, "Matrices vacías.")

    m = len(A)        # número de restricciones
    n = len(A[0])     # número de variables originales

    # Validaciones básicas
    if any(len(row) != n for row in A):
        return SimplexResult("infeasible", None, None, 0, "Filas de A con distinta longitud.")
    if len(b) != m:
        return SimplexResult("infeasible", None, None, 0, "Dimensión de b no coincide con A.")
    if len(c) != n:
        return SimplexResult("infeasible", None, None, 0, "Dimensión de c no coincide con A.")
    if any(bi < -1e-12 for bi in b):
        return SimplexResult("infeasible", None, None, 0,
                             "Este solver asume b >= 0 para base de holguras factible.")

    # Construir tableau con holguras: [A | I | b] y fila objetivo [-c | 0 | 0]
    # Variables: x (n), s (m). Base inicial: s.
    tableau = []
    for i in range(m):
        row = A[i][:] + [0.0]*m + [b[i]]
        row[n + i] = 1.0  # identidad para holguras
        tableau.append(row)

    # Fila objetivo (tenemos coeficientes negativos para maximizar)
    z_row = [-ci for ci in c] + [0.0]*m + [0.0]
    tableau.append(z_row)

    # Índices básicos (inicialmente holguras)
    basis = [n + i for i in range(m)]

    def print_tableau():
        if not show_steps:
            return
        print("\nTableau:")
        for r in tableau:
            print("  " + "  ".join(f"{v:10.4f}" for v in r))
        print(f"Base: {basis}\n")

    iterations = 0
    print_tableau()

    while True:
        iterations += 1
        # Columna entrante (regla de Bland): menor índice con coef < -eps en fila objetivo
        z = tableau[-1]
        entering_candidates = [j for j, val in enumerate(z[:-1]) if val < -1e-12]
        if not entering_candidates:
            # Óptimo alcanzado -> reconstruir x (solo variables originales)
            x = [0.0]*n
            for i, basic_var in enumerate(basis):
                if basic_var < n:
                    x[basic_var] = tableau[i][-1]
            z_opt = tableau[-1][-1]
            return SimplexResult("optimal", z_opt, x, iterations, "Óptimo encontrado.")

        entering = min(entering_candidates)  # Bland

        # Razón mínima para variable saliente
        ratios = []
        eligibles = []
        for i in range(m):
            col_val = tableau[i][entering]
            if col_val > 1e-12:
                ratios.append(tableau[i][-1] / col_val)
                eligibles.append(True)
            else:
                ratios.append(math.inf)
                eligibles.append(False)

        leaving = argmin_with_ties(ratios, eligibles)
        if leaving is None:
            return SimplexResult("unbounded", None, None, iterations,
                                 "La función objetivo puede crecer indefinidamente (no acotado).")

        # Pivoteo en (leaving, entering)
        pivot = tableau[leaving][entering]
        tableau[leaving] = [v / pivot for v in tableau[leaving]]  # normalizar fila pivote
        for i in range(m + 1):
            if i == leaving:
                continue
            factor = tableau[i][entering]
            if not approx_zero(factor):
                tableau[i] = [vi - factor * vj for vi, vj in zip(tableau[i], tableau[leaving])]

        # Actualizar base
        basis[leaving] = entering
        print_tableau()

def solve_and_print(A: List[List[float]], b: List[float], c: List[float], show_steps: bool = False):
    res = simplex_max(A, b, c, show_steps=show_steps)
    print("Estado:", res.status)
    print("Iteraciones:", res.iterations)
    print("Mensaje:", res.message)
    if res.status == "optimal":
        print("Valor óptimo z* =", round(res.z_opt, 6))
        print("x* =", [round(xi, 6) for xi in res.x_opt])

if __name__ == "__main__":
    # === PROBLEMA CARGADO ===
    # Max z = 3x1 + 5x2
    # s.a.
    #   2x1 +  x2 <= 8
    #    x1 + 2x2 <= 8
    #   x1, x2 >= 0
    A = [
        [2, 1],
        [1, 2],
    ]
    b = [8, 8]
    c = [3, 5]

    # Cambia a True si quieres ver el tableau paso a paso en la terminal
    SHOW_STEPS = False

    solve_and_print(A, b, c, show_steps=SHOW_STEPS)
