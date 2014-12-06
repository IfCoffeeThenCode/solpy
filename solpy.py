import math
import time

import numpy as np
import matplotlib.pyplot as plt

class body:
    def __init__(self, p, v, m, r):
        self.position, self.velocity, self.mass = np.array(p), np.array(v), m
        self.acceleration, self.radius, self.trace = np.zeros(3), r, None
        self.new_acceleration = None

def influence(b1, bodies, G):
    A = np.zeros(3)
    for b2 in bodies:
        if b2 != b1:
            R = b2.position - b1.position
            A += b2.mass * R / np.linalg.norm(R) ** 3
    return G * A

def simulate_leapfrog(bodies, steps, dt, G):
    init_trace(bodies, steps)
    for step in xrange(steps):
        add_trace(bodies, step)
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

def center_offset(bodies):
    C = np.zeros(3)
    V = np.zeros(3)
    M = 0
    for b in bodies:
        M += b.mass
        C += b.mass * b.position
        V += b.mass * b.velocity
    return - C / M, - V / M

def init_trace(bodies, steps):
    for b in bodies:
        b.trace = np.zeros(shape=(2, steps))

def add_trace(bodies, step):
    for b in bodies:
            b.trace[:, step] = b.position[:2]

def draw_traces(bodies, bare):
    for b in bodies:
        circle = plt.Circle((b.trace[0, 0], b.trace[1, 0]), radius=b.radius, fc='y')
        plt.gca().add_patch(circle)
    for b in bodies:
        plt.plot(b.trace[0], b.trace[1])
    plt.axis('scaled')
    if bare:
        plt.axis('off')
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
    plt.grid()
    plt.savefig("/home/dv/ipy/output-%s.png" % int(time.time()), dpi=300)
    plt.show()

def random_bodies():
    N = 10
    X = np.random.uniform(size=(N,3))
    V = np.random.uniform(size=(N,3))
    M = np.random.uniform(size=N)
    bodies = []
    for x, v, m in zip(X, V, M):
        bodies.append(body(x, v, m, 0.01))
    return 0.001, 1, bodies

def earth_and_moon():
    ship_phase = math.pi*1.47
    ship_orbit_height = 9e6
    ship_vel = 9320
    earth = body([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 5.98e+24, 6.378100e+6)
    moon = body([earth.position[0] + 3.84399e8, 0.0, 0.0], \
                [0.0, 1022 + earth.velocity[1], 0.0], 7.3477e22, 0.173814e6)

    bodies = [earth, moon]
    C, V = center_offset(bodies)
    earth.position += C
    earth.velocity += V

    ship = body([earth.position[0] + ship_orbit_height * math.cos(ship_phase), \
                 earth.position[1] + ship_orbit_height * math.sin(ship_phase), 0.0], \
                [earth.velocity[0] + ship_vel * math.cos(ship_phase + math.pi/2), \
                 earth.velocity[1] + ship_vel * math.sin(ship_phase + math.pi/2), 0.0], 1e8, 1e4)

    return 0.024 * 3600, 6.67428e-11, [earth, ship, moon]

def solar_system():
    dt = 24 * 3600
    G = 6.67428e-11

    sun = body([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.9891e+30, 696.000000e+6)
    mercury = body([0.57909175e+11, 0.0, 0.0], [0.0, 47872.5, 0.0], 3.302e+23, 2.439640e+6)
    venus = body([1.08208930e+11, 0.0, 0.0], [0.0, 35021.4, 0.0], 4.869e+24, 6.051590e+6)
    earth = body([1.49597871e+11, 0.0, 0.0], [0.0, 29800.0, 0.0], 5.98e+24, 6.378100e+6)
    mars = body([2.27936640e+11, 0.0, 0.0], [0.0, 24130.9, 0.0], 6.4191e+23, 3.397000e+6)
    jupiter = body([0.0, 7.40573600e+11 , 0.0], [-13720, 0.0, 0.0], 1.8986e+27, 71.492680e+6)
    saturn = body([-14.26725400e+11, 0.0, 0.0], [0.0, -9672.4, 0.0], 5.6851e+26, 60.267140e+6)
    uranus = body([0.0, -28.70972200e+11, 0.0], [6835.2, 0.0, 0.0], 8.6849e+24, 25.557250e+6)
    neptune = body([44.98252900e+11, 0.0, 0.0], [0.0, 5477.8, 0.0], 1.0244e+26, 24.766360e+6)

    bodies = [sun, mercury, venus, earth, mars, jupiter, saturn, uranus, neptune]
    C, V = center_offset(bodies)
    sun.position += C
    sun.velocity += V

    return dt, G, bodies

def main():
    dt, G, bodies = solar_system()
    simulate_leapfrog(bodies, 3*365, dt, G)
    draw_traces(bodies, False)

main()
