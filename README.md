# Sequential Quadratic Programming for Distance Minimization on Ellipsoidal Submanifold

## 1. Abstract

This repository presents a numerical framework for computing the metric projection of an external point onto an ellipsoidal submanifold in $\mathbb{R}^3$ via Sequential Quadratic Programming (SQP). The motivating scenario is a rescue capsule drifting in low Earth orbit whose onboard computer must continuously determine the closest point on Earth's surface to plan a re-entry trajectory. Earth is modelled as a triaxial ellipsoid, whose surface constitutes a smooth two-dimensional Riemannian submanifold embedded in $\mathbb{R}^3$. The framework demonstrates the complete numerical pipeline: formulating the projection as a constrained nonlinear program, deriving the Karush-Kuhn-Tucker optimality conditions analytically, constructing the exact Lagrangian Hessian, and solving the resulting saddle-point linear system iteratively. A warm-starting technique exploits the continuity of the orbital trajectory to reduce per-frame iteration counts from approximately ten to one or two, enabling real-time tracking.

---

## 2. Theoretical Background

### 2.1 The Ellipsoid as a Riemannian Submanifold

Let $a, b, c > 0$ be the three semi-axes. The ellipsoid $\mathcal{M}$ is defined as the zero-level set of the smooth function $g : \mathbb{R}^3 \to \mathbb{R}$:

```math
\mathcal{M} = \left\{ \mathbf{x} \in \mathbb{R}^3 \;\middle|\; g(\mathbf{x}) = \frac{x_1^2}{a^2} + \frac{x_2^2}{b^2} + \frac{x_3^2}{c^2} - 1 = 0 \right\}
```

By the Regular Level Set Theorem, $\mathcal{M}$ is a smooth embedded submanifold of dimension $n = 2$ in $\mathbb{R}^3$ if and only if $\nabla g(\mathbf{x}) \neq \mathbf{0}$ for all $\mathbf{x} \in \mathcal{M}$. Computing the gradient explicitly:

```math
\nabla g(\mathbf{x}) = \begin{bmatrix} \dfrac{2x_1}{a^2} \\ \dfrac{2x_2}{b^2} \\ \dfrac{2x_3}{c^2} \end{bmatrix}
```

This vector vanishes only at $\mathbf{x} = \mathbf{0}$, which does not lie on $\mathcal{M}$ since $g(\mathbf{0}) = -1 \neq 0$. Therefore $\mathcal{M}$ is a regular submanifold everywhere, and $\nabla g(\mathbf{x})$ serves as the outward normal direction at every surface point.

### 2.2 The Tangent Space

At any point $\mathbf{x} \in \mathcal{M}$, the tangent space $T_\mathbf{x}\mathcal{M}$ is the two-dimensional linear subspace of $\mathbb{R}^3$ orthogonal to the normal $\nabla g(\mathbf{x})$:

```math
T_\mathbf{x}\mathcal{M} = \ker\!\left(\nabla g(\mathbf{x})^T\right) = \left\{ \mathbf{v} \in \mathbb{R}^3 \;\middle|\; \frac{2x_1 v_1}{a^2} + \frac{2x_2 v_2}{b^2} + \frac{2x_3 v_3}{c^2} = 0 \right\}
```

Any admissible displacement along the surface must lie in this kernel. This is the geometric constraint that SQP enforces through linearization at each iteration.

### 2.3 The Second Fundamental Form and Curvature

The shape operator $S : T_\mathbf{x}\mathcal{M} \to T_\mathbf{x}\mathcal{M}$ encodes how the outward unit normal rotates as one moves along the surface. The second fundamental form $\mathbf{II}$ evaluated on tangent vectors $\mathbf{u}, \mathbf{v} \in T_\mathbf{x}\mathcal{M}$ is:

```math
\mathbf{II}(\mathbf{u}, \mathbf{v}) = \mathbf{u}^T \left( \frac{\nabla^2 g(\mathbf{x})}{\|\nabla g(\mathbf{x})\|} \right) \mathbf{v}
```

where the constraint Hessian is the constant diagonal matrix:

```math
\nabla^2 g(\mathbf{x}) = \begin{bmatrix} \dfrac{2}{a^2} & 0 & 0 \\ 0 & \dfrac{2}{b^2} & 0 \\ 0 & 0 & \dfrac{2}{c^2} \end{bmatrix}
```

The principal curvatures $\kappa_1, \kappa_2$ at a point $\mathbf{x}$ are the eigenvalues of the shape operator restricted to $T_\mathbf{x}\mathcal{M}$. For the ellipsoid, they vary continuously across the surface, where curvature is highest near the tips of the shortest semi-axis and smallest near the equatorial bulge. This spatial variation is precisely why the optimal Lagrange multiplier $\lambda^*$ is not constant: its magnitude encodes the local curvature of $\mathcal{M}$ at the projection point.

### 2.4 Metric Projection and Uniqueness

The metric projection of an external point $\mathbf{p} \notin \mathcal{M}$ onto $\mathcal{M}$ is defined as:

```math
\Pi_{\mathcal{M}}(\mathbf{p}) = \underset{\mathbf{x} \in \mathcal{M}}{\arg\min} \;\; \|\mathbf{x} - \mathbf{p}\|_2
```

Because the ellipsoid is a compact, strictly convex surface, every exterior point $\mathbf{p}$ has a unique closest point. The proof follows from strict convexity: the level sets of $f(\mathbf{x}) = \|\mathbf{x} - \mathbf{p}\|^2$ are spheres centred at $\mathbf{p}$, and a sphere can be tangent to a strictly convex surface at most at one point from the outside.

The signed distance function $d(\mathbf{p}) = \min_{\mathbf{x} \in \mathcal{M}} \|\mathbf{x} - \mathbf{p}\|$ is smooth everywhere except on the medial axis, which is the locus of points equidistant from two or more surface points. For the re-entry scenario, the capsule's orbital altitude ensures $\mathbf{p}$ remains well clear of the medial axis, guaranteeing smooth, unique projections throughout the trajectory.

### 2.5 Geometric Interpretation of the Lagrange Multiplier

The Lagrange multiplier $\lambda$ carries precise geometric meaning. The stationarity condition requires the displacement vector to be collinear with the outward surface normal. Any tangential component of this displacement would imply that sliding along $\mathcal{M}$ could reduce the distance, contradicting optimality. The scalar $\lambda$ is always negative for an exterior point, and its magnitude grows as the capsule descends toward the surface, reflecting the increasing curvature experienced by the approaching projection ray.

---

## 3. Mathematical Framework

### 3.1 The Constrained Nonlinear Program

We minimize the squared Euclidean distance, which shares its minimizer with the Euclidean distance itself but avoids a square root in the gradient computation:

```math
\min_{\mathbf{x} \in \mathbb{R}^3} \quad f(\mathbf{x}) = \|\mathbf{x} - \mathbf{p}\|^2 = (x_1 - p_1)^2 + (x_2 - p_2)^2 + (x_3 - p_3)^2
```

```math
\text{subject to} \quad g(\mathbf{x}) = \frac{x_1^2}{a^2} + \frac{x_2^2}{b^2} + \frac{x_3^2}{c^2} - 1 = 0
```

The gradients required at every SQP iteration are:

```math
\nabla f(\mathbf{x}) = 2(\mathbf{x} - \mathbf{p}) = \begin{bmatrix} 2(x_1 - p_1) \\ 2(x_2 - p_2) \\ 2(x_3 - p_3) \end{bmatrix}, \qquad \nabla g(\mathbf{x}) = \begin{bmatrix} \dfrac{2x_1}{a^2} \\ \dfrac{2x_2}{b^2} \\ \dfrac{2x_3}{c^2} \end{bmatrix}
```

### 3.2 The Lagrangian and KKT Conditions

For an equality-constrained nonlinear program, the constraint is incorporated into the objective via a Lagrange multiplier $\lambda \in \mathbb{R}$. The Lagrangian function $\mathcal{L} : \mathbb{R}^3 \times \mathbb{R} \to \mathbb{R}$ is:

```math
\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda \, g(\mathbf{x}) = \|\mathbf{x} - \mathbf{p}\|^2 + \lambda \left( \frac{x_1^2}{a^2} + \frac{x_2^2}{b^2} + \frac{x_3^2}{c^2} - 1 \right)
```

Expanding component-wise:

```math
\mathcal{L}(\mathbf{x}, \lambda) = (x_1 - p_1)^2 + (x_2 - p_2)^2 + (x_3 - p_3)^2 + \lambda\left(\frac{x_1^2}{a^2} + \frac{x_2^2}{b^2} + \frac{x_3^2}{c^2} - 1\right)
```

Setting $\partial \mathcal{L} / \partial x_i = 0$ for each coordinate yields the three stationarity equations:

```math
\frac{\partial \mathcal{L}}{\partial x_1} = 2(x_1 - p_1) + \frac{2\lambda x_1}{a^2} = 0 \quad \Longrightarrow \quad x_1 \left(1 + \frac{\lambda}{a^2}\right) = p_1
```

```math
\frac{\partial \mathcal{L}}{\partial x_2} = 2(x_2 - p_2) + \frac{2\lambda x_2}{b^2} = 0 \quad \Longrightarrow \quad x_2 \left(1 + \frac{\lambda}{b^2}\right) = p_2
```

```math
\frac{\partial \mathcal{L}}{\partial x_3} = 2(x_3 - p_3) + \frac{2\lambda x_3}{c^2} = 0 \quad \Longrightarrow \quad x_3 \left(1 + \frac{\lambda}{c^2}\right) = p_3
```

In compact vector form, this is the gradient stationarity condition:

```math
\nabla_\mathbf{x} \mathcal{L}(\mathbf{x}^*, \lambda^*) = \nabla f(\mathbf{x}^*) + \lambda^* \nabla g(\mathbf{x}^*) = \mathbf{0}
```

Stationarity with respect to $\lambda$ recovers the feasibility condition:

```math
\frac{\partial \mathcal{L}}{\partial \lambda} = g(\mathbf{x}^*) = \frac{(x_1^*)^2}{a^2} + \frac{(x_2^*)^2}{b^2} + \frac{(x_3^*)^2}{c^2} - 1 = 0
```

Together, these four scalar equations constitute the full nonlinear KKT system:

```math
\begin{bmatrix} 2(x_1^* - p_1) + \dfrac{2\lambda^* x_1^*}{a^2} \\ 2(x_2^* - p_2) + \dfrac{2\lambda^* x_2^*}{b^2} \\ 2(x_3^* - p_3) + \dfrac{2\lambda^* x_3^*}{c^2} \\ \dfrac{(x_1^*)^2}{a^2} + \dfrac{(x_2^*)^2}{b^2} + \dfrac{(x_3^*)^2}{c^2} - 1 \end{bmatrix} = \mathbf{0}
```

### 3.3 The Hessian of the Lagrangian and Second-Order Sufficiency

SQP requires a local quadratic model of the Lagrangian. The Hessian $\mathbf{H} = \nabla^2_{\mathbf{x}\mathbf{x}} \mathcal{L}(\mathbf{x}, \lambda)$ combines the constant objective Hessian $2\mathbf{I}$ with the scaled constraint Hessian:

```math
\mathbf{H} = 2\mathbf{I} + \lambda \begin{bmatrix} \dfrac{2}{a^2} & 0 & 0 \\ 0 & \dfrac{2}{b^2} & 0 \\ 0 & 0 & \dfrac{2}{c^2} \end{bmatrix} = \begin{bmatrix} 2 + \dfrac{2\lambda}{a^2} & 0 & 0 \\ 0 & 2 + \dfrac{2\lambda}{b^2} & 0 \\ 0 & 0 & 2 + \dfrac{2\lambda}{c^2} \end{bmatrix}
```

For the solution to be a strict local minimum, the Lagrangian Hessian projected onto the tangent space $T_{\mathbf{x}^*}\mathcal{M}$ must be positive definite:

```math
\mathbf{v}^T \mathbf{H}(\mathbf{x}^*, \lambda^*) \, \mathbf{v} > 0 \quad \forall \, \mathbf{v} \in T_{\mathbf{x}^*}\mathcal{M} \setminus \{\mathbf{0}\}
```

For any exterior point, $\lambda^* < 0$ and $|\lambda^*| < \min(a^2, b^2, c^2)$, so all diagonal entries of $\mathbf{H}$ remain strictly positive and the matrix is positive definite on all of $\mathbb{R}^3$. This guarantees the second-order sufficiency condition on the tangent space and confirms that every KKT point is the unique global minimizer.

---

## 4. Pipeline Architecture

The computational framework is structured as a sequential pipeline to evaluate the projection across continuous orbital trajectories.

| Phase | Process | Methodological Details |
| :--- | :--- | :--- |
| **1** | **Trajectory Generation** | Simulates the parametric orbital path $\mathbf{p}(t) \in \mathbb{R}^3$ of the rescue capsule over the defined time horizon. Provides a sequence of external query points for the projection solver. |
| **2** | **SQP Solver** | At each timestep, constructs the $4 \times 4$ KKT saddle-point system from the current iterate $(\mathbf{x}_k, \lambda_k)$, solves for the step $(\mathbf{d}, \Delta\lambda)$, and updates the state until $\|\mathbf{d}\|_2 \leq \epsilon$. |
| **3** | **Warm-Starting** | Injects the converged solution as the initial guess for the next timestep, exploiting trajectory continuity to minimise the number of required iterations per frame. |

### 4.1 Solving the KKT Subproblem

At each SQP iteration $k$, the nonlinear constraint is linearized about the current iterate $\mathbf{x}_k$:

```math
g(\mathbf{x}_k + \mathbf{d}) \approx g(\mathbf{x}_k) + \nabla g(\mathbf{x}_k)^T \mathbf{d} = 0
```

and the objective is approximated by a quadratic Taylor model:

```math
f(\mathbf{x}_k + \mathbf{d}) \approx f(\mathbf{x}_k) + \nabla f(\mathbf{x}_k)^T \mathbf{d} + \frac{1}{2} \mathbf{d}^T \mathbf{H}_k \mathbf{d}
```

Writing the KKT conditions of this local quadratic program yields the saddle-point linear system:

```math
\begin{bmatrix} \mathbf{H}_k & \nabla g(\mathbf{x}_k) \\ \nabla g(\mathbf{x}_k)^T & 0 \end{bmatrix} \begin{bmatrix} \mathbf{d} \\ \Delta\lambda \end{bmatrix} = \begin{bmatrix} -\nabla f(\mathbf{x}_k) - \lambda_k \nabla g(\mathbf{x}_k) \\ -g(\mathbf{x}_k) \end{bmatrix}
```

Substituting the explicit expressions, this becomes the fully expanded $4 \times 4$ system solved at each timestep:

```math
\begin{bmatrix} 2 + \frac{2\lambda_k}{a^2} & 0 & 0 & \frac{2x_1}{a^2} \\ 0 & 2 + \frac{2\lambda_k}{b^2} & 0 & \frac{2x_2}{b^2} \\ 0 & 0 & 2 + \frac{2\lambda_k}{c^2} & \frac{2x_3}{c^2} \\ \frac{2x_1}{a^2} & \frac{2x_2}{b^2} & \frac{2x_3}{c^2} & 0 \end{bmatrix} \begin{bmatrix} d_1 \\ d_2 \\ d_3 \\ \Delta\lambda \end{bmatrix} = \begin{bmatrix} -2(x_1 - p_1) - \frac{2\lambda_k x_1}{a^2} \\ -2(x_2 - p_2) - \frac{2\lambda_k x_2}{b^2} \\ -2(x_3 - p_3) - \frac{2\lambda_k x_3}{c^2} \\ -g(\mathbf{x}_k) \end{bmatrix}
```

The iterates are updated as:

```math
\mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{d}
```

```math
\lambda_{k+1} = \lambda_k + \Delta\lambda
```

Iterations continue until the step norm falls below the convergence tolerance $\epsilon > 0$:

```math
\|\mathbf{d}\|_2 = \sqrt{d_1^2 + d_2^2 + d_3^2} \leq \epsilon
```

### 4.2 Trajectory Tracking via Warm-Starting

The capsule follows a parametric orbital trajectory $\mathbf{p}(t) \in \mathbb{R}^3$ that changes smoothly between discrete timesteps. Since consecutive positions $\mathbf{p}(t)$ and $\mathbf{p}(t + \Delta t)$ are close, their projections onto $\mathcal{M}$ are also close. Rather than re-initializing SQP from scratch, the algorithm injects the previous converged solution as the initial guess:

```math
\mathbf{x}_{\text{init}}(t) = \mathbf{x}^*(t - \Delta t), \qquad \lambda_{\text{init}}(t) = \lambda^*(t - \Delta t)
```

The residual of the KKT system at the warm-start point is of order $\mathcal{O}(\left\| \frac{d\mathbf{p}}{dt} \right\| \Delta t)$, meaning the iterate is already close to the new solution. This reduces the required number of SQP iterations from approximately ten on a cold start to typically one or two, enabling real-time performance over long orbital arcs.

---

## 5. Limitations

| Limitation | Description |
| :--- | :--- |
| **Geometric Singularities** | If $\mathbf{p}$ lies exactly at the centre $\mathbf{0}$ of the ellipsoid, then $\nabla f(\mathbf{x}) = 2\mathbf{x}$ is parallel to $\nabla g(\mathbf{x})$ for every $\mathbf{x} \in \mathcal{M}$, and the KKT matrix becomes singular. Physically this would require the capsule to be at Earth's centre, which is not a realistic scenario. |
| **Matrix Inversion Cost** | The algorithm solves a $(D+1) \times (D+1)$ saddle-point system at every iteration. The $\mathcal{O}(D^3)$ cost of dense inversion is negligible for $D = 3$ but prohibitive for very high-dimensional manifolds. |
| **Local Optimality** | SQP converges to a local KKT point. For the ellipsoid, strict convexity guarantees this is the global minimizer for any exterior point. On non-convex surfaces the result depends on initialization. |
| **Warm-Start Drift** | If the capsule undergoes a rapid orbital correction manoeuvre, $\left\| \frac{d\mathbf{p}}{dt} \right\| \Delta t$ may be large and the warm-started iterate may lie far from the new solution, causing the iteration count to temporarily spike back to cold-start levels. |

---

## Getting Started

### Prerequisites
* Python 3.8+
* NumPy
* Matplotlib

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/mattia-3rne/sqp-submanifold-distance-minimization.git
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Analysis**:
    ```bash
    jupyter notebook
    ```

---

## Repository Structure

### Notebooks
* `main.ipynb`: The primary notebook containing the SQP implementation and trajectory visualizations.

### Source Code
* `sqp_solver.py`: Core logic for the KKT matrix construction and iterative step updates.
* `trajectory_generation.py`: Utilities for creating dynamic point paths in 3D space.

### Configuration & Data
* `requirements.txt`: Python dependencies. But obviously not as long of an only text introduction is needed and here
