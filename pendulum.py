import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class InvertedPendulum(object):
    M = 0.8
    m = 0.12
    l = 0.5
    g = 9.8
    t = 0.1
    t_num = 1000

    def __init__(self, x, theta, noisy=True):
        self.x = x
        self.x_dot = 0.
        self.theta = theta
        self.theta_dot = 0.
        self.u = 0.
        self.noisy = noisy
        self.t_one = self.t / self.t_num

    def do_action(self, u):

        self.u = u
        if self.noisy:
            self.u += np.random.uniform(-10, 10)

        self.update_state()
        return (self.theta, self.theta_dot)

    def update_state(self):
        for i in range(self.t_num):
            costheta = np.cos(self.theta)
            sintheta = np.sin(self.theta)
            ml = self.m * self.l
            total_mass = self.M + self.m

            temp = (self.u + ml * self.theta_dot**2 * sintheta) / total_mass
            thetaacc = ((self.g * sintheta - costheta * temp) /
                        (self.l * (4/3 - self.m * costheta**2 / total_mass)))
            xacc = temp - ml * thetaacc * costheta / total_mass

            self.x += self.t_one * self.x_dot
            self.x_dot += self.t_one * xacc
            self.theta += self.t_one * self.theta_dot
            self.theta_dot += self.t_one * thetaacc


    def get_car_x(self):
        return self.x


def video(x_history, angle_history, l, t):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    ax.grid()
    line, = ax.plot([], [], 'o-', lw=2)
    time_text = ax.text(0.02, 0.95, 'aaaaaa', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        line.set_data([x_history[i], x_history[i]+2*l*np.sin(angle_history[i])],
                      [0, 2*l*np.cos(angle_history[i])])
        time_text.set_text('time = {0:.1f}'.format(i*t))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, frames=range(len(x_history)),
                                  interval=1000*t, blit=False, init_func=init)
    ani.save("out.gif", writer = 'imagemagick')
    plt.show()


if __name__ == '__main__':
    plant = InvertedPendulum(0, np.pi/12, False)
    angle_history = [np.pi/12]
    x_history = [0.]

    u_list = [-2.0, 2.0]*50

    for i, u in enumerate(u_list):
        next_s = plant.do_action(u)
        print(next_s, next_s[0]*180/np.pi, plant.get_car_x())
        angle_history.append(next_s[0])
        x_history.append(plant.get_car_x())

    video(x_history, angle_history, plant.l, plant.t)