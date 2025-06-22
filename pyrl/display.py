# import gymnasium as gym
import imageio
import matplotlib.pyplot as plt


def render_state(env):
    image = env.render()
    plt.imshow(image)
    plt.show()


def create_gif(frames, filename="epsiode-result"):
    imageio.mimsave(f"{filename}.gif", frames, fps=5, loop=0)
