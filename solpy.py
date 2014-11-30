import math

import numpy as np
import matplotlib.pyplot as plt

class body:
    def __init__(self, p, v, m):
        self.position, self.velocity, self.mass = np.array(p), np.array(v), m
        self.trace_x, self.trace_y, self.acceleration = [], [], np.zeros(3)
        self.new_acceleration = np.zeros(3)

def influence(b1, bodies, G):
    A = 0
    for b2 in bodies:
        if b2 != b1:
            R = b2.position - b1.position
            A += b2.mass * R / np.linalg.norm(R) ** 3
    return G * A

def simulate_leapfrog(bodies, steps, dt, G):
    for i in xrange(steps):
        for b in bodies:
            b.trace_x.append(b.position[0])
            b.trace_y.append(b.position[1])
        for b in bodies:
            b.acceleration = influence(b, bodies, G)
        for b in bodies:
            b.position += b.velocity * dt + 1/2.0 * b.acceleration * dt ** 2
        for b in bodies:
            b.new_acceleration = influence(b, bodies, G)
        for b in bodies:
            b.velocity += 1/2.0 * (b.acceleration + b.new_acceleration) * dt
        for b in bodies:
            b.acceleration = b.new_acceleration

def draw_traces(bodies):
    for b in bodies:
        plt.plot(b.trace_x, b.trace_y)
    plt.axis('scaled')
    plt.show()

def toy():
    b1 = body([0.0, 0, 0], [0.0, 0.0, 0], 10)
    b2 = body([-1.0, 1.0, 0], [1.7, 0, 0], 0.001)
    b3 = body([1.0, -1.0, 0], [-1.7, 0, 0], 0.001)
    return 0.01, 1, [b1, b2, b3]

def solar_system():
    dt = 24 * 3600
    G = 6.67428e-11

    ship_phase = math.pi*1.53
    ship_orbit_height = 1e8
    ship_vel = 2516.265

    sun = body([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.9891e+30)
    mercury = body([0.57909175e+11, 0.0, 0.0], [0.0, 47872.5, 0.0], 3.302e+23)
    venus = body([1.08208930e+11, 0.0, 0.0], [0.0, 35021.4, 0.0], 4.869e+24)
    earth_sol = body([1.49597871e+11, 0.0, 0.0], [0.0, 29800.0, 0.0], 5.98e+24)
    earth_alone = body([1.49597871e+11, 0.0, 0.0], [0.0, 0.0, 0.0], 5.98e+24)
    earth = earth_sol
    moon = body([earth.position[0] + 3.84399e8, 0.0, 0.0], \
                [0.0, 1022 + earth.velocity[1], 0.0], 7.3477e22)
    mars = body([2.27936640e+11, 0.0, 0.0], [0.0, 24130.9, 0.0], 6.4191e+23)
    ship = body([earth.position[0] + ship_orbit_height * math.cos(ship_phase), \
                 earth.position[1] + ship_orbit_height * math.sin(ship_phase), 0.0], \
                [earth.velocity[0] + ship_vel * math.cos(ship_phase + math.pi/2), \
                 earth.velocity[1] + ship_vel * math.sin(ship_phase + math.pi/2), 0.0], 1e6)
    jupiter = body([7.78357721e+11, 0.0, 0.0], [0.0, 13100, 0.0], 1.9e+27)
    saturn = body([14.26725400e+11, 0.0, 0.0], [0.0, 9672.4, 0.0], 5.6851e+26)
    uranus = body([28.70972200e+11, 0.0, 0.0], [0.0, 6835.2, 0.0], 8.6849e+24)
    neptune = body([44.98252900e+11, 0.0, 0.0], [0.0, 5477.8, 0.0], 1.0244e+26)

    return dt, G, [sun, mercury, venus, earth, moon, mars, jupiter, saturn, uranus, neptune]

def main():
    dt, G, bodies = solar_system()
    simulate_leapfrog(bodies, 300, dt, G)
    draw_traces(bodies)

main()