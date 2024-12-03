from RobotV2_1 import genetic_algorithm, Robot


def simulate_battles():
        # Get the best and worst robots
        best_robots, worst_robots = genetic_algorithm()

        # Battle between the best robots
        print("==================================================")
        print("Best battle:")
        Robot.battle(best_robots[0], best_robots[1])
        print("==================================================")

        # Battle between the worst robots
        print("==================================================")
        print("Worst battle:")
        Robot.battle(worst_robots[0], worst_robots[1])
        print("==================================================")


simulate_battles()
