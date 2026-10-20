# Faculty Review-1 Notes → What's Implemented Where

| Faculty note | Status | Where |
|---|---|---|
| Interaction doesn't scale linearly | Implemented | `identification/edmdc.py`, demonstrated on Lotka-Volterra and Drake data |
| System identification — Subspace ID / SINDy | SINDy implemented; Subspace ID covered in report text only | `identification/sindy.py`; `docs/report_sections.md` §6 |
| Higher dimension / Koopman / nonlinear->linear | Implemented | `identification/edmdc.py`, `docs/report_sections.md` §2 |
| 4 angle constant — pitch, 6 DOF | Confirmed | Drake's `QuadrotorPlant` 12-state = 6-DOF rigid body |
| Add noise — simulate the reality | Implemented | `identification/kalman_filter.py::inject_noise`, compared in `compare_methods.py` |
| Kalman filtering | Implemented | `identification/kalman_filter.py` |
| Is the covariance varying? | Addressed (constant used, limitation flagged) | `docs/report_sections.md` §5 |
| Pick a topic, explain deeply | Koopman/EDMDc + SINDy chosen as the deep-dive topic | All of `identification/` + `docs/report_sections.md` |
| Circular trajectory | Implemented on Drake | `drake_sim/circle_tracking.py` |
| Visible simulation on the Drake platform | Implemented via Meshcat + QuadrotorGeometry | `drake_sim/circle_tracking.py` |
| Meshgrid visualization | Implemented | `identification/meshgrid_viz.py` |

## Honest notes on scope

- Subspace Identification (N4SID) is covered as a literature comparison in
  the report text, not implemented in code.
- `meshgrid_viz.py` Part B (Drake roll-pitch field) requires
  `drake_sim/circle_tracking.py` to have been run first (needs saved data).
