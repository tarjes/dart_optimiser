import numpy as np

class DartBoard:
    def __init__(self):
        # board dimensions in mm (measured from center of board)
        self.r1 = 6.35      # distance to double bullseye
        self.r2 = 15.9      # distance to single bullseye
        self.r3 = 99.0      # distance to inside triple ring
        self.r4 = 107.0     # distance to outside triple ring
        self.r5 = 162.0     # distance to inside double ring
        self.r = 170.0      # distance to outside double ring

        self.numbers = [6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10]
        self.half_numbers = [6, 13, 13, 4, 4, 18, 18, 1, 1, 20, 20, 5, 5, 12, 12, 9, 9, 14, 14, 11, 11, 8, 8, 16, 16, 7,
                            7, 19, 19, 3, 3, 17, 17, 2, 2, 15, 15, 10, 10, 6]

        self.width = np.pi / 10
        self.half_width = np.pi / 20.0

        self.regions_enum = ['Bull', '1x', '2x', '3x', '1xi']

    @staticmethod
    def xy_to_polar(x, y):
        # Theta must be within +- 360 degrees
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        return r, theta

    @staticmethod
    def polar_to_xy(r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    def theta_to_number(self, theta):
        return self.half_numbers[int(theta // self.half_width)]

    def polar_to_score(self, r, theta):
        if r <= self.r1:
            return 50
        elif r <= self.r2:
            return 25
        elif r <= self.r3:
            return self.theta_to_number(theta)
        elif r <= self.r4:
            return 3 * self.theta_to_number(theta)
        elif r <= self.r5:
            return self.theta_to_number(theta)
        elif r <= self.r:
            return 2 * self.theta_to_number(theta)
        else:
            return 0

    def xy_to_score(self, x, y):
        r, theta = self.xy_to_polar(x, y)
        return self.polar_to_score(r, theta)

    def get_aim_polar(self, number, region):
        if region == 0 or region == 'bullseye':
            return 0, 0
        elif region == 1 or region == 'outer_single':
            r = (self.r5 + self.r4) / 2
        elif region == 2 or region == 'double':
            r = (self.r + self.r5) / 2
        elif region == 3 or region == 'tripple':
            r = (self.r4 + self.r3) / 2
        elif region == 4 or region == 'inner_single':
            r = (self.r4 + self.r5) / 2
        else:
            raise Exception("Region must be 0, 1, 2, 3 or 4")

        # Find index number from number list
        i = self.numbers.index(number)
        theta = float(i) * self.width

        return r, theta

    def get_aim(self, number, region):
        r, theta = self.get_aim_polar(number, region)
        return self.polar_to_xy(r, theta)

    def hit_to_score(self, hit):
        return self.xy_to_score(hit[0], hit[1])


class DartArm:

    ## Sigma guide ##
    # 170 - USELESS - misses the board most of the time
    # 100 - BAD - hits the board most of the time, the rest is pure luck

    # 10 - Bullseye 1 on 3


    def __init__(self, x_sigma=80, y_sigma=80):
        self.x_sigma = x_sigma
        self.y_sigma = y_sigma

    def set_sigma(self, x_sigma, y_sigma):
        self.x_sigma = x_sigma
        self.y_sigma = y_sigma


    def throw(self, aim):
        x_aim = aim[0]
        y_aim = aim[1]
        x_res = np.random.normal(x_aim, self.x_sigma)
        y_res = np.random.normal(y_aim, self.y_sigma)
        return x_res, y_res


def simulate_n_throws(board, arm, aim, n):
    res = []
    for i in range(n):
        res.append(board.hit_to_score(arm.throw(aim)))
    return res

def simulate_game(board, arm, aim):
    l = simulate_n_throws(board, arm, aim, 3)
    l.remove(min(l))
    return sum(l)

def simulate_n_games(board, arm, aim, n):
    res = []
    for i in range(n):
        res.append(simulate_game(board, arm, aim))
    return res


def print_sorted_dict(dict1):
    for key in sorted(dict1, key=lambda k: dict1[k], reverse=True):
        print(key, ': ', dict1[key])

def best_aim_by_average(board, arm, n):
    res = {}
    # Bull is bull
    average_score = sum(simulate_n_games(board, arm, (0,0), n)) / n
    res[f'{board.regions_enum[0]}:  '] = average_score
    for region in range(1, 5):
        for number in board.numbers:
            aim = board.get_aim(number, region)
            average_score = sum(simulate_n_games(board, arm, aim, n))/n
            res[f'{board.regions_enum[region]} {number}'] = average_score
            #print(f'{average_score}: \t\t\t\t {board.regions_enum[region]} {number}')

    print_sorted_dict(res)
    return res

def best_aim_by_beat_score(board, arm, goal, n):
    res = {}
    # Bull is bull
    games = simulate_n_games(board, arm, (0,0), n)
    count_above_goal = sum(i >= goal for i in games)
    res[f'{board.regions_enum[0]}:  '] = count_above_goal
    for region in range(1, 5):
        for number in board.numbers:
            aim = board.get_aim(number, region)

            games = simulate_n_games(board, arm, aim, n)
            count_above_goal = sum(i >= goal for i in games)

            res[f'{board.regions_enum[region]} {number}'] = count_above_goal
    print_sorted_dict(res)
    return res


ret = best_aim_by_average(DartBoard(), DartArm(30,30), 1000)

# IDEAS

# 1000 simulations is not enough

# Use pandas to analyse simulation results

# Add a region 5, the single between bull and tripple

# Optimize code for large simulations

# Throwing first:
# Should give out 3 (or more) different strategies
# - Safe (your opponent is worse than you)
# - Highest average (normal)
# - Aggressive (your opponent is better than you)

# Throwing last:
# - I NEED TO BEAT THIS SUM


# The average 2/3 sum might not be a good measure. Or.. if you are playing a worse opponent, you can aim for
# the distribution with high average and low spread. If you are facing someone better than you, you may have to gamble more.
# Visual sigma setter
# Automatic sigma-detection