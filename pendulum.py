import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class InvertedPendulum(object):
    M = 0.7
    m = 0.12
    l = 0.3
    J = 9e-3
    g = 9.8
    t = 0.1
    t_num = 120

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
        ml = self.m * self.l
        Mt = self.M + self.m
        Jt = self.J + ml*self.l

        for i in range(self.t_num):
            costheta = np.cos(self.theta)
            sintheta = np.sin(self.theta)
            alpha = Mt*Jt - (ml*costheta)**2

            thetaacc = ( ml*sintheta*( Mt*self.g - ml*self.theta_dot**2*costheta ) - self.u*ml*costheta ) / alpha
            xacc = - ( ml*sintheta*( ml*self.g*costheta - Jt*self.theta_dot**2 ) - self.u*Jt ) / alpha

            self.x += self.t_one * self.x_dot
            self.x_dot += self.t_one * xacc
            self.theta += self.t_one * self.theta_dot
            self.theta_dot += self.t_one * thetaacc

    def get_car_x(self):
        return self.x

    def get_state(self):
        return self.x, self.theta, self.x_dot, self.theta_dot


def video(x_history, angle_history, l, t, animation_save=False):
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

    if animation_save:
        ani.save("./test_out.gif", writer = 'imagemagick')
    else:
        plt.show()


if __name__ == '__main__':
    plant = InvertedPendulum(0, np.pi/12, False)
    angle_history = [np.pi/12]
    x_history = [0.]

    u_list = [-2.0, 2.0]*50

    for i, u in enumerate(u_list):
        next_s = plant.do_action(u)
        #print(next_s, next_s[0]*180/np.pi, plant.get_car_x())
        angle_history.append(next_s[0])
        x_history.append(plant.get_car_x())

    print(angle_history)
    video(x_history, angle_history, plant.l, plant.t)