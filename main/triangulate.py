import numpy as np

if __name__ == "__main__":
    
    np.set_printoptions(precision=7, suppress=True, linewidth=100)

    #Take in inputs
    distance = float(input("Enter the distance of the waypoint: "))
    x = float(input("Enter x: "))
    z = float(input("Enter z: "))
    angle = float(input("Enter angle: "))

    #Convert to radians from deg
    angle = (angle * np.pi)/180

    #x = rcos(theta)
    x_coord = distance * np.cos(angle)
    #z = rsin(theta)
    z_coord = distance * np.sin(angle)

    print(x_coord + x, z_coord + z)
