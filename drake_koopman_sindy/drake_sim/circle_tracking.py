"""
Drake Quadrotor — Circle Tracking with Live Meshcat Visualization
===================================================================
- Loads Drake's built-in QuadrotorPlant (6-DOF rigid body: 12-state).
- Computes a linearized LQR controller around hover.
- Tracks a circular reference trajectory.
- Visualizes live in Meshcat (open the printed URL in your browser).
- Logs (state, input) snapshots to ../data/ for downstream identification.

Run:  python3 circle_tracking.py
Then open the printed http://localhost:7000 URL in your browser BEFORE
the simulation finishes advancing (it visualizes as it runs).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from pydrake.systems.framework import DiagramBuilder, LeafSystem, BasicVector
from pydrake.systems.analysis import Simulator
from pydrake.systems.primitives import LogVectorOutput, Linearize
from pydrake.systems.controllers import LinearQuadraticRegulator
from pydrake.examples import QuadrotorPlant, QuadrotorGeometry
from pydrake.geometry import StartMeshcat, SceneGraph

# Visualization helpers differ slightly by Drake version; try the modern path first.
try:
    from pydrake.visualization import AddDefaultVisualization
    HAVE_DEFAULT_VIS = True
except ImportError:
    HAVE_DEFAULT_VIS = False

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Hover linearization + LQR gain
# ----------------------------------------------------------------------
Z0 = 1.0
R_CIRCLE = 0.5
W_CIRCLE = 0.5

plant_for_lin = QuadrotorPlant()
ctx_for_lin = plant_for_lin.CreateDefaultContext()

x_hover = np.zeros(12)
x_hover[2] = Z0
ctx_for_lin.SetContinuousState(x_hover)

m = plant_for_lin.m()
g = 9.81
u_hover = np.full(4, m * g / 4)
plant_for_lin.get_input_port(0).FixValue(ctx_for_lin, u_hover)

Q = np.diag([10., 10., 10., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
R_cost = np.eye(4) * 0.1

lin = Linearize(plant_for_lin, ctx_for_lin)
K, S = LinearQuadraticRegulator(lin.A(), lin.B(), Q, R_cost)
print(f"[Drake] hover mass={m:.4f} kg | LQR gain K shape={K.shape}")


class CircleTrackingController(LeafSystem):
    """u = u_hover - K @ (x_current - x_ref(t)), x_ref traces a circle."""

    def __init__(self, K):
        LeafSystem.__init__(self)
        self.K = K
        self.DeclareVectorInputPort("x", BasicVector(12))
        self.DeclareVectorOutputPort("u", BasicVector(4), self.CalcControl)

    def CalcControl(self, context, output):
        t = context.get_time()
        x_ref = np.zeros(12)
        x_ref[0] = R_CIRCLE * np.cos(W_CIRCLE * t)
        x_ref[1] = R_CIRCLE * np.sin(W_CIRCLE * t)
        x_ref[2] = Z0
        x_current = np.array(self.get_input_port(0).Eval(context))
        u = u_hover - self.K @ (x_current - x_ref)
        output.SetFromVector(u)


# ----------------------------------------------------------------------
# 2. Build the diagram (plant + controller + visualization + loggers)
# ----------------------------------------------------------------------
meshcat = StartMeshcat()
print(f"[Drake] Meshcat running — open the URL printed above in your browser.")

builder = DiagramBuilder()
plant = builder.AddSystem(QuadrotorPlant())
controller = builder.AddSystem(CircleTrackingController(K))

builder.Connect(controller.get_output_port(0), plant.get_input_port(0))
builder.Connect(plant.get_output_port(0), controller.get_input_port(0))

logger_x = LogVectorOutput(plant.get_output_port(0), builder)
logger_u = LogVectorOutput(controller.get_output_port(0), builder)

# ---- Real quadrotor mesh geometry (not just a marker) ----
# QuadrotorGeometry is Drake's purpose-built helper that wires a SceneGraph
# and the actual quadrotor visual mesh to a QuadrotorPlant's state output --
# this is what Drake's own official quadrotor demo uses, so it's the
# lower-risk path to a real rendered drone body (vs. rebuilding on
# MultibodyPlant+URDF from scratch).
vis_attached = False
if HAVE_DEFAULT_VIS:
    try:
        scene_graph = builder.AddSystem(SceneGraph())
        QuadrotorGeometry.AddToBuilder(builder, plant.get_output_port(0), scene_graph)
        AddDefaultVisualization(builder=builder, meshcat=meshcat)
        vis_attached = True
        print("[Drake] QuadrotorGeometry attached -- you should see a real drone "
              "mesh (not just a marker) in the Meshcat browser tab.")
    except Exception as e:
        print(f"[Drake] NOTE: QuadrotorGeometry/AddDefaultVisualization failed -- {e}")
        print("[Drake] Falling back to a manual Meshcat marker showing live position.")

diagram = builder.Build()
simulator = Simulator(diagram)
sim_context = simulator.get_mutable_context()
plant_context = plant.GetMyMutableContextFromRoot(sim_context)

x0 = np.zeros(12)
x0[0] = R_CIRCLE
x0[2] = Z0
plant_context.SetContinuousState(x0)

# ----------------------------------------------------------------------
# 3. Run the simulation.
#    If the real QuadrotorGeometry mesh attached successfully (vis_attached),
#    Meshcat will already animate it live as AdvanceTo() runs -- no manual
#    marker needed. If it failed for any reason, fall back to a manual
#    marker so you still get SOME live visible motion in the browser.
# ----------------------------------------------------------------------
simulator.Initialize()
simulator.set_target_realtime_rate(1.0)  # run at real-world speed so Meshcat playback is watchable

T_FINAL = 30.0
DT_VIS = 0.05
t = 0.0

if not vis_attached:
    from pydrake.geometry import Sphere, Rgba
    from pydrake.math import RigidTransform
    meshcat.SetObject("drone_marker", Sphere(0.05), rgba=Rgba(0.1, 0.4, 0.9, 1.0))
    while t < T_FINAL:
        t_next = min(t + DT_VIS, T_FINAL)
        simulator.AdvanceTo(t_next)
        state_now = plant.get_output_port(0).Eval(plant.GetMyContextFromRoot(sim_context))
        pos = np.array(state_now[0:3])
        meshcat.SetTransform("drone_marker", RigidTransform(pos))
        t = t_next
else:
    # Step in small increments so Meshcat's realtime playback stays visible
    # as the sim runs, rather than jumping straight to t=30.
    while t < T_FINAL:
        t_next = min(t + DT_VIS, T_FINAL)
        simulator.AdvanceTo(t_next)
        t = t_next

print("[Drake] Simulation complete.")

# ----------------------------------------------------------------------
# 4. Extract logs, plot, save data for downstream identification
# ----------------------------------------------------------------------
log_x = logger_x.FindLog(sim_context)
log_u = logger_u.FindLog(sim_context)
states = log_x.data()   # (12, N)
inputs = log_u.data()   # (4, N)

plt.figure()
plt.plot(states[0, :], states[1, :])
plt.xlabel("x position (m)")
plt.ylabel("y position (m)")
plt.title("Drake drone circle tracking")
plt.axis("equal")
plt.savefig(os.path.join(PLOT_DIR, "drake_circle_result.png"))
plt.close()

X = states[:, :-1]
Xprime = states[:, 1:]
U = inputs[:, :-1]
np.save(os.path.join(DATA_DIR, "X_drake.npy"), X)
np.save(os.path.join(DATA_DIR, "Xprime_drake.npy"), Xprime)
np.save(os.path.join(DATA_DIR, "U_drake.npy"), U)

print(f"[Drake] Saved snapshots: X{X.shape}, Xprime{Xprime.shape}, U{U.shape}")
print(f"[Drake] Plot saved to plots/drake_circle_result.png")
print("[Drake] NOTE: keep the Meshcat browser tab open WHILE this script runs "
      "to see the live marker trace the circle in 3D.")
