from pathlib import Path
import mujoco


MODEL_PATH = Path(__file__).with_name("quadrotor.xml")


def main():
    print("=" * 60)
    print("MUJOCO QUADROTOR MODEL INSPECTION")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    print(f"Model loaded successfully.")
    print()

    print("Model dimensions")
    print("-----------------")
    print(f"nq  = {model.nq}")
    print(f"nv  = {model.nv}")
    print(f"nu  = {model.nu}")
    print(f"nbody = {model.nbody}")
    print(f"njnt  = {model.njnt}")
    print()

    print("Expected:")
    print("nq = 7   -> position + quaternion")
    print("nv = 6   -> linear + angular velocity")
    print("nu = 4   -> four motors")
    print()

    print("Actuators")
    print("---------")

    for i in range(model.nu):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            i,
        )

        print(f"Motor {i + 1}: {name}")

    print()

    print("Body IDs")
    print("--------")

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "quadrotor",
    )

    print("quadrotor body ID:", body_id)

    print()
    print("MODEL INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()