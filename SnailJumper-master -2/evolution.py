import copy
import numpy as np
from player import Player



class Evolution:
    def __init__(self):
        self.game_mode= "Neuroevolution"
        self.generation_number = 1

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """

        sorted_players = sorted(players, key=lambda player_selected: player_selected.fitness)
        sorted_players = sorted_players[-num_players:]

        f = open("sorted_player.txt", "a")
        f.write(str(self.generation_number) + ':'
                + str(sorted_players[0].fitness) + ':' + str(sorted_players[-1].fitness) + ':'
                + str(int(np.mean([my_player.fitness for my_player in sorted_players]))) + ',')
        f.close()

        self.generation_number += 1

        return sorted_players

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            selected_parents = []
            children = []
            for player in prev_players:
                selected_parents.append(self.clone_player(player))
            for i in range(0, len(selected_parents), 2):
                child1 = Player(self.game_mode)
                child2 = Player(self.game_mode)

                size_child_w1 = int(child1.nn.w1.shape[0] / 2)
                size_child_w2 = int(child1.nn.w2.shape[0] / 2)
                size_child_b2 = int(child1.nn.b2.shape[0] / 2)
                size_child_b3 = int(child1.nn.b3.shape[0] / 2)

                child1.nn.w1[0:size_child_w1, :] = \
                    selected_parents[i].nn.w1[0:size_child_w1, :]
                child2.nn.w1[0:size_child_w1, :] = \
                    selected_parents[i + 1].nn.w1[0:size_child_w1, :]
                child1.nn.w1[size_child_w1:, :] = \
                    selected_parents[i + 1].nn.w1[size_child_w1:, :]
                child2.nn.w1[size_child_w1:, :] = \
                    selected_parents[i].nn.w1[size_child_w1:, :]

                child1.nn.w2[0:size_child_w2, :] = \
                    selected_parents[i].nn.w2[0:size_child_w2, :]
                child2.nn.w2[0:size_child_w2, :] = \
                    selected_parents[i + 1].nn.w2[0:size_child_w2, :]
                child1.nn.w2[size_child_w2:, :] = \
                    selected_parents[i + 1].nn.w2[size_child_w2:, :]
                child2.nn.w2[size_child_w2:, :] = \
                    selected_parents[i].nn.w2[size_child_w2:, :]

                child1.nn.b2[0:size_child_b2, :] = \
                    selected_parents[i].nn.b2[0:size_child_b2, :]
                child2.nn.b2[0:size_child_b2, :] = \
                    selected_parents[i + 1].nn.b2[0:size_child_b2, :]
                child1.nn.b2[size_child_b2:, :] = \
                    selected_parents[i + 1].nn.b2[size_child_b2:, :]
                child2.nn.b2[size_child_b2:, :] = \
                    selected_parents[i].nn.b2[size_child_b2:, :]

                child1.nn.b3[0:size_child_b3, :] = \
                    selected_parents[i].nn.b3[0:size_child_b3, :]
                child2.nn.b3[0:size_child_b3, :] = \
                    selected_parents[i + 1].nn.b3[0:size_child_b3, :]
                child1.nn.b3[size_child_b3:, :] = \
                    selected_parents[i + 1].nn.b3[size_child_b3:, :]
                child2.nn.b3[size_child_b3:, :] = \
                    selected_parents[i].nn.b3[size_child_b3:, :]

                children.append(child1)
                children.append(child2)

            for child in children:
                self.mutate(child)
        return children
    def mutate(self, child):

        # child: an object of class `Player`
        random_number_array = np.random.uniform(low=0, high=1, size=4)
        if random_number_array[0] < 0.2:
            child.nn.w1 = child.nn.w1 + np.random.normal(size=child.nn.w1.shape)
        if random_number_array[1] < 0.2:
            child.nn.w2 = child.nn.w2 + np.random.normal(size=child.nn.w2.shape)
        if random_number_array[2] < 0.2:
            child.nn.b2 = child.nn.b2 + np.random.normal(size=child.nn.b2.shape)
        if random_number_array[3] < 0.2:
            child.nn.b3 = child.nn.b3 + np.random.normal(size=child.nn.b3.shape)

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player

    def Qtournament(self, players, q):
        q_selected = np.random.choice(players, q)
        return max(q_selected, key=lambda player: player.fitness)
