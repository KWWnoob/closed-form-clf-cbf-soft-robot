from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve

import jax
from jax import Array
import jax.numpy as jnp
from cbfpy.cbfs.clf_cbf import CLFCBF, CLFCBFConfig

import matplotlib.pyplot as plt
import numpy as onp
from functools import partial

# Newer soromox API path
from soromox.systems import TendonActuatedPCS

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from viz.open3d_vis_V2 import visualize_robot_open3d
from viz.matplot_vis import animate_robot_tendons_matplotlib  # noqa: F401

jax.config.update("jax_enable_x64", True)

jnp.set_printoptions(
    threshold=jnp.inf,
    linewidth=jnp.inf,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)
jnp.set_printoptions(precision=4, suppress=True)


def make_robot(num_segments: int, params: dict) -> TendonActuatedPCS:
    """Build the robot in a way that is compatible with newer soromox versions.

    The current environment appears to expose `TendonActuatedPCS` from
    `soromox.systems`, and newer versions often expect
    `active_tendon_routing_params` instead of `tendon_routing_params`.
    """
    r0, r1 = float(params["r"][0]) - 0.005, float(params["r"][1]) - 0.005
    theta0 = jnp.deg2rad(jnp.array([120, 240, 0]))
    theta1 = jnp.deg2rad(jnp.array([180, 300, 60]))
    ry0, rz0 = r0 * jnp.cos(theta0), r0 * jnp.sin(theta0)
    ry1, rz1 = r1 * jnp.cos(theta1), r1 * jnp.sin(theta1)

    tendon_routing_params = {
        "ry": jnp.concatenate([ry0, ry1]),
        "rz": jnp.concatenate([rz0, rz1]),
        "my": jnp.zeros(6),
        "mz": jnp.zeros(6),
        "idx_seg_att": jnp.array([0, 0, 0, 1, 1, 1]),
    }

    # Try the newer API first.
    try:
        robot = TendonActuatedPCS(
            num_segments=num_segments,
            params=params,
            active_tendon_routing_params=tendon_routing_params,
            order_gauss=5,
        )
    except TypeError:
        # Fallback for older APIs, if needed.
        strain_selector = (
            jnp.array([True, True, True, True, True, True])[None, :]
            .repeat(num_segments, axis=0)
            .flatten()
        )
        robot = TendonActuatedPCS(
            num_segments=num_segments,
            params=params,
            tendon_routing_params=tendon_routing_params,
            strain_selector=strain_selector,
            order_gauss=5,
        )

    if int(robot.num_actuators) <= 0:
        q_test = jnp.zeros((int(robot.num_active_strains),), dtype=jnp.float64)
        try:
            a_shape = robot.actuation_matrix(q_test).shape
        except Exception as exc:  # pragma: no cover
            a_shape = f"actuation_matrix failed: {exc}"
        raise ValueError(
            "Robot was created but has no active actuators. "
            f"num_active_strains={robot.num_active_strains}, "
            f"num_actuators={robot.num_actuators}, A_shape={a_shape}. "
            "Your soromox build likely expects a different tendon-routing API."
        )

    return robot


@jax.jit
def pairwise_h(
    p: jnp.ndarray,
    centers: jnp.ndarray,
    r_obs: jnp.ndarray,
    r_robot: jnp.ndarray | float = 0.0,
    safety: float = 0.00,
) -> jnp.ndarray:
    """Return pairwise distance-based barrier functions."""
    p = jnp.asarray(p)
    centers = jnp.asarray(centers)
    r_obs = jnp.asarray(r_obs)

    r_robot = jnp.asarray(r_robot)
    r_robot = r_robot + jnp.zeros((p.shape[0],), dtype=p.dtype)

    diff = p[:, None, :] - centers[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1)
    H = dist - (r_obs[None, :] + r_robot[:, None] + safety)
    return H


if __name__ == "__main__":
    num_segments = 2
    rho = 1070 * jnp.ones((num_segments,))

    params = {
        "p0": jnp.array([jnp.pi / 2, jnp.pi / 2, 0.0, 0.0, 0.0, 0.0]),
        "L": 15e-2 * 2 / num_segments * jnp.ones((num_segments,)),
        "r": 3.6e-2 * jnp.ones((num_segments,)),
        "rho": rho,
        "g": jnp.array([0.0, 0.0, 9.81]),
        "E": 20e3 * jnp.ones((num_segments,)),
        "G": 20e3 * jnp.ones((num_segments,)),
    }

    params["D"] = 1e-3 * jnp.diag(
        (
            jnp.repeat(
                jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]), num_segments, axis=0
            )
            * params["L"][:, None]
        ).flatten()
    )

    robot = make_robot(num_segments, params)

    print("num_active_strains:", robot.num_active_strains)
    print("num_actuators:", robot.num_actuators)

    class TendonSoroConfig(CLFCBFConfig):
        def __init__(self):
            self.p_d_2 = jnp.array([0.10, 0.05, 0.32])
            self.s_ps = jnp.linspace(0, sum(params["L"]), 20 * num_segments)
            self.robot_radius = params["r"][0]
            self.safety_margin = 0.00

            self.obs = {
                "centers": jnp.array([
                    [0.10, 0.08, 0.24],
                    [0.12, 0.06, 0.32],
                    [0.04, 0.055, 0.20],
                ]),
                "radii": jnp.array([0.02, 0.02, 0.02]),
            }

            self.J_ltheta = jnp.diag(0.045 * jnp.ones(int(robot.num_actuators)))

            super().__init__(
                n=int(robot.num_active_strains),
                m=int(robot.num_actuators),
                relax_cbf=False,
                cbf_relaxation_penalty=1e6,
                clf_relaxation_penalty=10,
            )

        def f(self, z: Array) -> Array:
            return jnp.zeros_like(z)

        def g(self, z: Array) -> Array:
            # Velocity-level kinematic mapping.
            J_lq = robot.actuation_matrix(z).T
            return jnp.linalg.pinv(J_lq)

        def V_1(self, z: Array) -> Array:
            g_ee_2 = robot.forward_kinematics(z, jnp.sum(robot.L))
            p_2 = g_ee_2[:3, 3]
            e_2 = jnp.linalg.norm(p_2 - self.p_d_2[:3])
            return e_2.reshape(-1)

        def h_1(self, z: Array, kappa: float = 2000.0) -> Array:
            g_ee = robot.forward_kinematics_batched(z, self.s_ps)
            p = g_ee[:, :3, 3]
            H = pairwise_h(
                p,
                self.obs["centers"],
                self.obs["radii"],
                self.robot_radius,
                self.safety_margin,
            )
            h_all = (-1.0 / kappa) * jnp.log(jnp.sum(jnp.exp(-kappa * H)))
            return h_all.reshape(-1)

        def alpha_1(self, h_1: Array) -> Array:
            return h_1 * 10.0

        def gamma_1(self, V_1: Array) -> Array:
            return V_1 * 10.0

    config = TendonSoroConfig()
    clf_cbf = CLFCBF.from_config(config)
    z_des = config.p_d_2

    @jax.jit
    def control_input(q: Array, z_des: Array) -> Array:
        # Prefer the closed-form controller the user wanted; fallback to the
        # generic controller if the installed cbfpy version lacks this method.
        if hasattr(clf_cbf, "controller_closed_form_hard_clf_then_cbf_I"):
            u = clf_cbf.controller_closed_form_hard_clf_then_cbf_I(q, z_des)
        else:
            u = clf_cbf.controller(q, z_des)
        return u

    @jax.jit
    def closed_loop_ode_fn(t: float, q: Array, z_des: Array) -> Array:
        del t
        u = control_input(q, z_des)
        J_lq = robot.actuation_matrix(q).T
        q_dot = jnp.linalg.pinv(J_lq) @ u
        return q_dot

    q0 = jnp.repeat(
        jnp.array([0.0, 0.0, 0.0 * jnp.pi, 0.0, 0.0, 0.0])[None, :],
        num_segments,
        axis=0,
    ).flatten()

    t0 = 0.0
    t1 = 4.0
    dt = 1e-4

    sol = diffeqsolve(
        ODETerm(closed_loop_ode_fn),
        Tsit5(),
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=q0,
        args=(z_des,),
        saveat=SaveAt(ts=jnp.arange(t0, t1, dt)),
        max_steps=None,
    )

    ts, q_ts = sol.ts, sol.ys

    # Clearance plot.
    g_ee_tsp = jax.vmap(lambda q: robot.forward_kinematics_batched(q, config.s_ps))(q_ts)
    p_tsp = g_ee_tsp[:, :, :3, 3]
    centers = config.obs["centers"]
    radii = config.obs["radii"]

    dist_tpN = jnp.linalg.norm(
        p_tsp[:, :, None, :] - centers[None, None, :, :],
        axis=-1,
    ) - (radii[None, None, :] + config.robot_radius)

    min_clearance_tN = dist_tpN.min(axis=1)
    global_min_t = onp.asarray(min_clearance_tN).min(axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(onp.asarray(ts), global_min_t)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Time [s]")
    plt.ylabel("Global min clearance [m]")
    plt.title("Global minimum clearance over time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # End-effector trajectory plot.
    forward_kinematics_end_effector = jax.jit(
        partial(robot.forward_kinematics, s=jnp.sum(robot.L))
    )
    g_ee_ts = jax.vmap(forward_kinematics_end_effector)(q_ts)

    plt.figure()
    plt.plot(ts, g_ee_ts[:, 0, 3], label="End-effector x [m]")
    plt.plot(ts, g_ee_ts[:, 1, 3], label="End-effector y [m]")
    plt.plot(ts, g_ee_ts[:, 2, 3], label="End-effector z [m]")
    plt.xlabel("Time [s]")
    plt.ylabel("End-effector position [m]")
    plt.legend()
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()

    def obs_for_vis(obs_dict, colors=None):
        centers = jnp.asarray(obs_dict["centers"])
        radii = jnp.asarray(obs_dict["radii"])
        n_obs = len(radii)
        if colors is None:
            colors = [(0.5, 0.5, 0.5)] * n_obs
        assert len(colors) == n_obs, "Number of colors must match number of obstacles"
        return [(tuple(centers[i]), float(radii[i]), tuple(colors[i])) for i in range(n_obs)]

    stride = 200
    visualize_robot_open3d(
        robot,
        t_list=ts[::stride],
        q_list=q_ts[::stride],
        num_points=80,
        fps=60,
        target_point=config.p_d_2,
        target_radius=0.01,
        target_color=(1.0, 0.1, 0.1),
        obstacles=obs_for_vis(config.obs),
        record={"dir": "frames", "every_n": 1},
    )

    # Save CSV.
    ts_onp = onp.asarray(ts)
    qts_onp = onp.asarray(q_ts)
    data_to_save = onp.column_stack([ts_onp, qts_onp])
    header = "time," + ",".join([f"q{i}" for i in range(qts_onp.shape[1])])

    onp.savetxt(
        "setpoint_results_new.csv",
        data_to_save,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.8f",
    )

    print("✅ Saved to setpoint_results_new.csv")
