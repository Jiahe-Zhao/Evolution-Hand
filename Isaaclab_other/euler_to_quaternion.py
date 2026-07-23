import math


def euler_to_quaternion(roll, pitch, yaw):
    """
    Converts Euler angles (roll, pitch, yaw) to a quaternion (w, x, y, z).

    Args:
        roll (float): Rotation around the X-axis in radians.
        pitch (float): Rotation around the Y-axis in radians.
        yaw (float): Rotation around the Z-axis in radians.

    Returns:
        tuple: Quaternion as (w, x, y, z).
    """
    # Compute trigonometric values
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # Compute quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return (w, x, y, z)


# Example usage:
roll = math.radians(0)  # Convert degrees to radians if needed
pitch = math.radians(0)
yaw = math.radians(0)

quaternion = euler_to_quaternion(roll, pitch, yaw)
print("Quaternion:", quaternion)
