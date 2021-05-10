import matplotlib.pyplot as plt
import sys

sys.path.append(r"../../")
import blockweek_ad.ca_utils.tools as to


styles = plt.style.available
wp = to.load_course_data("../database/town04-waypoints.csv")
course1 = to.load_course_data("../database/course1.csv")
course2 = to.load_course_data("../database/course2.csv")

colors = [(252 / 255, 63 / 255, 82 / 255),    # red
          (255 / 255, 166 / 255, 106 / 255),  # orange
          (0 / 255, 183 / 255, 194 / 255),    # teal/cyan
          '#08F7FE',
          '#FE53BB',  # pink
          '#F5D300',  # yellow
          '#00ff41'   # matrix green
          ]


def course_full():
    fig, ax = plt.subplots()

    ax.set_ylim([-650, 520])
    ax.set_ylim([-50, 480])

    plt.gca().invert_yaxis()
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_facecolor((31 / 255, 28 / 255, 31 / 255))
    ax.scatter(wp[:, 0], wp[:, 1], color=(60 / 255, 60 / 255, 60 / 255), marker='.')

    ax.plot(course1[:, 0], course1[:, 1], color='gray', linewidth=5)
    ax.plot(course1[:, 0], course1[:, 1], color='black', linewidth=3)
    ax.plot(course2[:, 0], course2[:, 1], color='gray', linewidth=5)
    ax.plot(course2[:, 0], course2[:, 1], color='black', linewidth=3)

    ax.plot(course1[:, 0], course1[:, 1], color=(255 / 255, 166 / 255, 106 / 255), marker='o', markersize=10,
            linewidth=5, alpha=0.05)
    ax.plot(course1[:, 0], course1[:, 1], color=(255 / 255, 102 / 255, 0 / 255), marker='.', markersize=2,
            linewidth=2, alpha=1)  # (152/255, 3/255, 18/255)

    ax.plot(course2[:, 0], course2[:, 1], color=(255 / 255, 166 / 255, 106 / 255), marker='o', markersize=10,
            linewidth=5, alpha=0.05)
    ax.plot(course2[:, 0], course2[:, 1], color=(255 / 255, 102 / 255, 0 / 255), marker='.', markersize=2,
            linewidth=2, alpha=1)  # (152/255, 3/255, 18/255)

    ax.text(0, 360, str('FH DORTMUND RACE'), fontfamily='barlow', fontsize=22, fontweight='normal', color=(255 / 255, 102 / 255, 0 / 255))
    ax.text(0, 367, str('TOWN 04'), fontfamily='barlow', fontsize=15, fontweight='ultralight', color='white')


def course1_animation():
    fig, ax = plt.subplots()
    step = 5
    for idx in range(0, len(course1) + step, step):
        ax.cla()

        ax.set_ylim([-650, 520])
        ax.set_ylim([-50, 480])

        plt.gca().invert_yaxis()
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_facecolor((31/255, 28/255, 31/255))
        ax.scatter(wp[:, 0], wp[:, 1], color=(60/255, 60/255, 60/255), marker='.')

        ax.plot(course1[:, 0], course1[:, 1], color='gray', linewidth=5)
        ax.plot(course1[:, 0], course1[:, 1], color='black', linewidth=3)
        ax.plot(course2[:, 0], course2[:, 1], color='gray', linewidth=5)
        ax.plot(course2[:, 0], course2[:, 1], color='black', linewidth=3)

        n_lines = 10
        diff_linewidth = 1.05
        alpha_value = 0.03
        for n in range(1, n_lines + 1):
            ax.plot(course1[:idx, 0], course1[:idx, 1], marker='o',
                    linewidth=2 + (diff_linewidth * n),
                    alpha=alpha_value,
                    color=(252 / 255, 63 / 255, 82 / 255))

        ax.text(100, 330, str('COURSE 1'), fontfamily='barlow', fontsize=15, fontweight='ultralight',
                color=(252 / 255, 88 / 255, 104 / 255))
        ax.text(100, 360, str('FH DORTMUND RACE'), fontfamily='barlow', fontsize=22, fontweight='normal', color='white')
        ax.text(100, 380, str('TOWN 04'), fontfamily='barlow', fontsize=15, fontweight='ultralight', color='white')

        if idx == 0:
            plt.pause(3)
        else:
            plt.pause(0.00000001)

        plt.savefig("gif1/{0:4d}.png".format(idx))


def course2_animation():
    fig, ax = plt.subplots()
    step = 5
    for idx in range(0, len(course1) + step, step):
        ax.cla()

        ax.set_ylim([-650, 520])
        ax.set_ylim([-50, 480])

        plt.gca().invert_yaxis()
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_facecolor((31/255, 28/255, 31/255))
        ax.scatter(wp[:, 0], wp[:, 1], color=(60/255, 60/255, 60/255), marker='.')

        ax.plot(course1[:, 0], course1[:, 1], color='gray', linewidth=5)
        ax.plot(course1[:, 0], course1[:, 1], color='black', linewidth=3)
        ax.plot(course2[:, 0], course2[:, 1], color='gray', linewidth=5)
        ax.plot(course2[:, 0], course2[:, 1], color='black', linewidth=3)

        n_lines = 10
        diff_linewidth = 1.05
        alpha_value = 0.03
        for n in range(1, n_lines + 1):
            ax.plot(course2[:idx, 0], course2[:idx, 1], marker='o',
                    linewidth=2 + (diff_linewidth * n),
                    alpha=alpha_value,
                    color='#08F7FE')

        ax.text(100, 330, str('COURSE 2'), fontfamily='barlow', fontsize=15, fontweight='ultralight',
                color=(1/255, 173/255, 165/255))
        ax.text(100, 360, str('FH DORTMUND RACE'), fontfamily='barlow', fontsize=22, fontweight='normal', color='white')
        ax.text(100, 380, str('TOWN 04'), fontfamily='barlow', fontsize=15, fontweight='ultralight', color='white')

        if idx == 0:
            plt.pause(3)
        else:
            plt.pause(0.00000001)

        plt.savefig("gif2/{0:4d}.png".format(idx))


def course_animation(course_id, color, step=1):
    course = to.load_course_data("../database/course{}.csv".format(course_id))
    fig, ax = plt.subplots()

    for idx in range(0, len(course) + step, step):
        ax.cla()

        ax.set_ylim([-650, 520])
        ax.set_ylim([-50, 480])

        plt.gca().invert_yaxis()
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_facecolor((31/255, 28/255, 31/255))
        ax.scatter(wp[:, 0], wp[:, 1], color=(60/255, 60/255, 60/255), marker='.')

        ax.plot(course1[:, 0], course1[:, 1], color='gray', linewidth=5)
        ax.plot(course1[:, 0], course1[:, 1], color='black', linewidth=3)
        ax.plot(course2[:, 0], course2[:, 1], color='gray', linewidth=5)
        ax.plot(course2[:, 0], course2[:, 1], color='black', linewidth=3)

        n_lines = 10
        diff_linewidth = 1.05
        alpha_value = 0.03
        for n in range(1, n_lines + 1):
            ax.plot(course[:idx, 0], course[:idx, 1], marker='o',
                    linewidth=2 + (diff_linewidth * n),
                    alpha=alpha_value,
                    color=color)

        ax.text(100, 330, "COURSE {}".format(course_id), fontfamily='barlow', fontsize=15, fontweight='ultralight',
                color=color)
        ax.text(100, 360, str('FH DORTMUND RACE'), fontfamily='barlow', fontsize=22, fontweight='normal', color='white')
        ax.text(100, 380, str('TOWN 04'), fontfamily='barlow', fontsize=15, fontweight='ultralight', color='white')

        if idx == 0:
            plt.pause(3)
        else:
            plt.pause(0.00000001)


# course_animation(8, 'red', step=10)
course_full()
plt.show()
