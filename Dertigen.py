import random as rand
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pygame
import time
from tqdm import tqdm
rand.seed(int(time.time()))

import os
from typing import List, Callable, Tuple

# Lines 17-211 is player class, wordt gebruikt voor beide simulatie en gui
# Lines 212 - 519 zijn de simulatie, als je die wil gebruiken moeten er een paar "#" worden weggehaald
# Lines 533 - 1036 is de GUI, eerst classes, daarna de game loop, in de game loop zie je welke events kunnen gebeuren en hoe


# IDEA: use fixed reward for each ending of dice as you are not dependent on what kind of player you are up against


# GENERAL TO DO:  
# 1. Add doc comments to each class and method
# 2. Refactor longer methods
# 3. Split game part and Simulation/RL agent part

# TO DO: reorganize this class
class Player:
    def __init__(self, id: int, strategy: str, alpha: float, beta: float, is_simulation: bool = False, use_doubling: bool= False):
        self._agent = None
        if (strategy == "QLearner"):
            self._agent = Agent(
                gamma=0.9,
                epsilon_init=1.0, 
                epsilon_decay=0.0001, 
                epsilon_final=0.01, 
                alpha=0.01,
                id=id,
                use_doubling = use_doubling
            )
        self._id = id # 1-indexed
        self._strategy = strategy
        self._stripes = 0
        self._times_drunk = 0
        self._stripes_given = 0
        self._doubling_count = 0
        self._times_one_first = 0
        self._best_doubling = 0
        self._best_extra_above_30 = 0
        self._max_drunk = 0
        self._threshold = 15
        self._utilities = []
        self._alpha = alpha
        self._beta = beta
        self._is_simulation = is_simulation
        self._pending_stripes = 0


    def reset(self) -> None:
        self._stripes = 0
        self._times_drunk = 0
        self._stripes_given = 0
        self._doubling_count = 0
        self._times_one_first = 0
        self._best_doubling = 0
        self._best_extra_above_30 = 0
        self._max_drunk = 0
        
    
    @property
    def stripes(self) -> int:
        return self._stripes

    @stripes.setter
    def stripes(self, value: int) -> None:
        self._stripes = value

    @property
    def times_drunk(self) -> int:
        return self._times_drunk

    @times_drunk.setter
    def times_drunk(self, value: int) -> None:
        self._times_drunk = value

    @property
    def stripes_given(self) -> int:
        return self._stripes_given

    @stripes_given.setter
    def stripes_given(self, value: int) -> None:
        self._stripes_given = value

    @property
    def strategy(self) -> str:
        return self._strategy

    @strategy.setter
    def strategy(self, value: str) -> None:
        self._strategy = value

    @property
    def id(self) -> int:
        return self._id
    
    @property
    def doubling_count(self) -> int:
        return self._doubling_count
    
    @doubling_count.setter
    def doubling_count(self, value: int) -> None:
        self._doubling_count = value

    @property
    def times_one_first(self) -> int:
        return self._times_one_first
    
    @times_one_first.setter
    def times_one_first(self, value: int) -> None:
        self._times_one_first = value

    @property 
    def best_doubling(self) -> int:
        return self._best_doubling
    
    @best_doubling.setter
    def best_doubling(self, value: int) -> None:
        self._best_doubling = value

    @property
    def best_extra_above_30(self) -> int:
        return self._best_extra_above_30
    
    @best_extra_above_30.setter
    def best_extra_above_30(self, value: int) -> None:
        self._best_extra_above_30 = value

    @property
    def max_drunk(self) -> int:
        return self._max_drunk
    
    @property
    def utilities(self) -> list:
        return self._utilities
    
    
    @property
    def alpha(self) -> float:
        return self._alpha
    
    @property
    def beta(self) -> float:
        return self._beta
    
    @property 
    def is_simulation(self) -> bool:
        return self._is_simulation
    
    @property
    def pending_stripes(self) -> int:
        return self._pending_stripes
    
    @property
    def agent(self) -> 'Agent':
        return self._agent

    def add_utility(self, utility: float) -> None:
        self._utilities.append(utility)
    
    def set_utility(self, utility: float, index: int) -> None:
        if index >= len(self._utilities):
            print("This player does not exist, try to change another player's utility!")
        else:
            self._utilities[index] = utility
    
    def get_own_utility(self) -> float:
        return self._utilities[self._id-1]
    
    def set_own_utility(self, utility: float) -> None:
        self._utilities[self._id-1] = utility

    def add_stripes_given(self, extra_stripes: int) -> None:
        self._stripes_given += extra_stripes

    def add_stripes(self, extra_stripes: int) -> None:
        if self._is_simulation:
            self._stripes += extra_stripes
        else:
            self._pending_stripes += extra_stripes  

    def update_stripes(self) -> None:
        if self._pending_stripes > 0:
            for base in [1000, 100, 10, 1]:
                if self._pending_stripes >= base:
                    self._stripes += base  
                    self._pending_stripes -= base  
                    break  
        elif self._pending_stripes < 0:
            for base in [1000, 100, 10, 1]:
                if self._pending_stripes <= -base:
                    self._stripes -= base  
                    self._pending_stripes += base  
                    break  

    def double_stripes(self) -> None:
        self.add_stripes(self._stripes)

    def drink(self) -> int:
        times_to_drink = self._stripes // self._threshold
        self._max_drunk = max(self._max_drunk, times_to_drink)
        self._times_drunk += times_to_drink
        self.add_stripes(-times_to_drink * 15)
        return times_to_drink

    def get_given_taken_ratio(self) -> float:
        return self._stripes_given / self._times_drunk if self._times_drunk != 0 else 0
    
    def get_doubling_success_rate(self) -> float:
        return self._doubling_count / (self._times_one_first) if (self._times_one_first) != 0 else 0
    
    # New methods for updating statistics
    def increment_doubling_count(self) -> None:
        self._doubling_count += 1
    
    def increment_times_one_first(self) -> None:
        self._times_one_first += 1
    
    def update_best_doubling(self, value: int) -> None:
        self._best_doubling = max(self._best_doubling, value)
    
    def update_best_extra_above_30(self, value: int) -> None:
        self._best_extra_above_30 = max(self._best_extra_above_30, value)

    def get_columns(self):
        return ["Name", "Stripes", "Times drunk", "Double count", "Stripes given"]

    def get_row(self):
        return [self._strategy, self._stripes, self._times_drunk, self._doubling_count, self._stripes_given]

class Agent:
    def __init__(self,
                epsilon_init: float,
                epsilon_decay: float,
                epsilon_final: float,
                alpha: float,
                gamma: float,
                id: int,
                use_doubling: bool):
        self.q_values = [[0] * 37 for _ in range(7)]

        self.epsilon = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_final = epsilon_final
        
        self.learning_rate = alpha
        self.discount_factor = gamma
        self.id = id
        self.use_doubling = use_doubling 

        self.training_errors = []

    def get_action(self, freq: list[int], cur_sum: int):
        valid_indices = [i for i, x in enumerate(freq) if x != 0] 
        if (np.random.random() < self.epsilon):
            return rand.choice(valid_indices)
        else:
            best_idx = valid_indices[-1]
            dice_count = DICE - sum(freq)
            for i in valid_indices:
                count = freq[i]
                contender_val = self.q_values[dice_count + count][cur_sum + count * (i+1)]
                best_val = self.q_values[dice_count + freq[best_idx]][cur_sum + freq[best_idx] * (best_idx+1)]
                if contender_val > best_val:
                    best_idx = i
            return best_idx
          
    
    def update(self,
               cur_sum: int,
               freq: list[int],
               reward: float,
               terminated: bool,
               next_freq: list[int]):
    
        max_q_value = 0
        dice_chosen = DICE - sum(freq)
        if not terminated:
            for idx, count in enumerate(next_freq):
                if count > 0:
                    max_q_value = max(max_q_value, self.q_values[dice_chosen + count][cur_sum + count *(idx+1)])

        temporal_difference = reward + self.discount_factor*max_q_value - self.q_values[dice_chosen][cur_sum] # rightside of formula, difference in value
        self.q_values[dice_chosen][cur_sum] += self.learning_rate * temporal_difference # you weight the difference by alpha, the learning rate
        self.training_errors.append(temporal_difference) # you add the difference to a list, so you can see how much the policy changes
    
    def decay(self):
        self.epsilon = max(self.epsilon_final, self.epsilon - self.epsilon_decay)
        




REPLICATIONS = 1000000
DICE = 6
SIDES = 6

PLAYERCOUNT = int(input("How many players do you want in this game? "))
# IS_SIMULATION = input("Do you want to run a simulation? (True/False) ")
# IS_SIMULATION = True if IS_SIMULATION == "True" else False
IS_SIMULATION = True

# Initialize tracking dictionaries for each player
dice_choice_freq_per_player = [{i: 0 for i in range(1, SIDES + 1)} for _ in range(PLAYERCOUNT)]
ending_sum_freq_per_player = [{i: 0 for i in range(DICE * SIDES + 1)} for _ in range(PLAYERCOUNT)]


# these lists are constructed by a simulation in java with 1e8 replications for each entry
underTenProbabilityMemo = [
    [0.2768436, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.3128557, 0.2027189, 0.1186102, 0.0551224, 0.0209805, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.3677238, 0.2431631, 0.1469158, 0.0704483, 0.0280604, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.4503896, 0.3123747, 0.1959137, 0.0988955, 0.0420171, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.5929544, 0.4258245, 0.2963749, 0.1573801, 0.074112, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.8334398, 0.6667124, 0.500127, 0.3333399, 0.1667368]
    ]
expectedEndSumMemo = [
    [29.26835475, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 24.72776855, 25.72860164, 26.72927068, 27.73020606, 28.72966293, 29.65602928, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 20.25839557, 21.25863885, 22.25829802, 23.25783515, 24.2582571, 25.25855278, 26.25873597, 27.2581612, 28.25836748, 29.25793823, 30.16084095, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 15.97092354, 16.9707134, 17.9709818, 18.9707325, 19.97119338, 20.97096539, 21.9714933, 22.97076772, 23.97106191, 24.97067244, 25.97059233, 26.97066911, 27.97093096, 28.97109286, 29.97070267, 30.82008243, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 11.97192252, 12.97187984, 13.97156643, 14.97218331, 15.97178991, 16.97211489, 17.97248859, 18.97234921, 19.97239492, 20.97216985, 21.97185248, 22.97166245, 23.97182865, 24.97234685, 25.97183191, 26.97182376, 27.97173144, 28.97225384, 29.9723622, 30.97222574, 31.63870702, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 8.4997123, 9.49973745, 10.50035223, 11.50032722, 12.50014574, 13.49972308, 14.49997945, 15.5002137, 16.49993284, 17.49987203, 18.49978675, 19.50010456, 20.49977317, 21.5002023, 22.49988549, 23.49981693, 24.5000758, 25.50002119, 26.49998706, 27.50008943, 28.49982686, 29.5000993, 30.49999669, 31.49965572, 32.49988215]
]
expectedStripesMemo = [
    [1.68206039, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 5.28243674, 4.34184413, 3.46153276, 2.66391809, 1.97649455, 1.42299438, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 9.64767212, 8.7024032, 7.71641827, 6.72291387, 5.72513305, 4.76971219, 3.8401925, 2.98713789, 2.22963051, 1.59676582, 1.10491311, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 9.31464475, 11.37412322, 11.84825913, 10.90529189, 9.92834958, 9.02918346, 8.02916715, 7.02894929, 6.02965979, 5.02915512, 4.07103526, 3.16988502, 2.36579664, 1.67718012, 1.12723805, 0.73829532, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 7.1658865, 9.92525216, 11.81385213, 13.74922749, 13.47234195, 13.02803964, 12.0275384, 11.02763392, 10.02752508, 9.02776954, 8.02811801, 7.02754449, 6.02782585, 5.02764732, 4.0275967, 3.10142035, 2.25907402, 1.55518348, 0.98151193, 0.57404464, 0.34269577, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 3.16590901, 6.16706347, 8.99797207, 11.66655587, 14.16581425, 16.49998159, 15.4997544, 14.49968483, 13.50001704, 12.49978578, 11.49979552, 10.50002263, 9.50020448, 8.49990452, 7.50003166, 6.49988023, 5.50003433, 4.50028031, 3.49995469, 2.49996531, 1.66678298, 0.99990981, 0.49997542, 0.16665148, 0.0]
]
expectedGivenStripesMemo = [
    [2.664744038979394, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.05895985120614905, 0.2134380540875069, 0.5443338024770046, 1.1094505557739025, 1.983404392854119, 3.0189908993773127, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07856136540532278, 0.2758300297870447, 0.687132909503231, 1.3645351044083762, 2.391893164273342, 3.5430504076838947, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11716957579169424, 0.3936941009591141, 0.9414043964553556, 1.8135618585916713, 3.072939123560742, 4.360880217539916, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20715720582750594, 0.647479038488219, 1.4763257129695433, 2.6687250324436556, 4.327304286349778, 5.544683639056445, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.466316353103476, 1.3993724775367364, 2.7986301664261246, 4.664306265081332, 6.995551219979035]
]
expectedGivenStripesMemoAbove30 = [2.798420295491079, 5.596840590982158, 8.395260886473237, 11.193681181964315, 13.992101477455394, 16.790521772946473]

# TO DO: either remove these utilities or organize them better
def linear_utility(x: float) -> float:
    return x

def exponential_utility(x: float, a: float):
    return (math.exp(a * x) - 1) / (math.exp(a) - 1) if a != 0 else (math.exp(a * x) - 1)

def x_log_x_utility(x: float) -> float:
    return  x * math.log(1+x) 

def x_sqrt_x_utility(x: float) -> float:
    utility = x * math.sqrt(x)
    return  utility

def draw_dice(dice_count: int) -> list[int]:
    """Used to get new action space
    
    Args:
        dice_count: number of dice to throw
    """
    frequency = [0] * SIDES
    for _ in range(dice_count):
        frequency[rand.randint(0, SIDES-1)] += 1
    return frequency

def handle_dice(dice_count: int, cur_sum: int, index: int, freq: int, id: int, dice_choice_freq_per_player: list[dict[int, int]]) -> tuple[int, int]:
    """Function used to take an action with index+1 being the dice chosen,
    outputs the new dice count and new sum after picking these die.
    
    Args:
        dice_count: number of dice not yet taken
        cur_sum: current sum of die taken
        index: index in freq of the die chosen (0-indexed)
        freq: frequency of the die chosen
        dice_choice_freq_per_player: list of dictionaries used to capture the stats per player"""
    dice_count -= freq
    dice_choice_freq_per_player[id-1][index+1] += freq
    cur_sum += freq * (index+1)
    return dice_count, cur_sum

def risk_strategy(frequency: list, dice_count: int, cur_sum: int, id: int, dice_choice_freq_per_player: list[dict[int, int]], player_list: list[Player]) -> tuple[int, int]:
    for index, freq in enumerate(frequency):
        if freq > 0:
            if dice_count == 6:
                player_list[id-1].increment_times_one_first()
            dice_count, cur_sum = handle_dice(dice_count, cur_sum, index, freq, id, dice_choice_freq_per_player)
            break 
    return dice_count, cur_sum

def risk_averse_strategy(frequency: list, dice_count: int, cur_sum: int, id: int, dice_choice_freq_per_player: list[dict[int, int]]) -> tuple[int, int]:
    for index in range(len(frequency) - 1, -1, -1):
        freq = frequency[index]
        if freq > 0:
            [dice_count, cur_sum] = handle_dice(dice_count, cur_sum, index, freq, id, dice_choice_freq_per_player)
            break
    return dice_count, cur_sum

def personal_strategy(frequency: list, dice_count: int, cur_sum: int, id: int, dice_choice_freq_per_player: list[dict[int,int]], player_list: list[Player]) -> tuple[int, int]:
    print(f"Your have currently thrown {DICE - dice_count} dice and you have a sum of {cur_sum}")
    index = int(input(f"Here are the frequencies of the dice: {frequency}, what dice do you choose to hold? "))-1
    for i in range(SIDES):
        if (frequency[i] > 0):
            if i == index:
                player_list[id-1].increment_times_one_first
            break
    dice_count, cur_sum = handle_dice(dice_count, cur_sum, index, frequency[index], id, dice_choice_freq_per_player)
    return dice_count, cur_sum

# TO DO: Refactor this method
def prepare_smart_risktaker(frequency, dice_count, cur_sum) -> tuple[list[int], list[int], list[float]]:
    expected_stripes_for_self = [0] * 6
    expected_given_stripes_above_30 = [0] * 6
    p_doubling = [0] * 6

    for index, freq in enumerate(frequency):
        if freq > 0:
            temp_dice_count = dice_count - freq
            temp_cur_sum = cur_sum + freq * (index+1)
            # cases: 1. thrown all 6 dice -> you know the outcomes now, so compare real with expected as usual  2. sum 30 still one dice -> just throw it, and this is obviously the best option 3. Normal (still have dice to throw under 30) -> use memo tables
            # case1:
            if temp_dice_count == 0:
                expected_stripes_for_self[index] = 30 - temp_cur_sum if (10 < temp_cur_sum < 30) else 0
                expected_given_stripes_above_30[index] = expectedGivenStripesMemoAbove30[temp_cur_sum - 31] if temp_cur_sum > 30 else 0
                p_doubling[index] = 0.0 if temp_cur_sum > 10 else 1.0  
            #case2:    
            elif temp_dice_count == 1 and temp_cur_sum == 30:
                expected_stripes_for_self[index] = 0
                expected_given_stripes_above_30[index] = (expectedGivenStripesMemoAbove30[2] + expectedGivenStripesMemoAbove30[3]) / 2
                p_doubling[index] = 0.0
            #case3:
            else:
                expected_stripes_for_self[index] = expectedStripesMemo[DICE - temp_dice_count][temp_cur_sum]
                expected_given_stripes_above_30[index] = expectedGivenStripesMemo[DICE-temp_dice_count][temp_cur_sum]
                p_doubling[index] = underTenProbabilityMemo[DICE - temp_dice_count][temp_cur_sum] if temp_cur_sum < 10 else 0.0
                # expectedStripesForSelf[index] *= (1-p_doubling[index])

    return expected_stripes_for_self, expected_given_stripes_above_30, p_doubling

# TO DO: Refactor this method
def best_choice_smart_risktaker(frequency: list, expected_stripes_for_self: list[int], expected_given_stripes_above_30: list[int], p_doubling: list[float], player_list: list[Player], id: int) -> tuple:
    best_score = float('-inf') # initialize with a low value
    best_index = 0
    best_freq = frequency[best_index]
    cur_player = player_list[id-1]
    # Idea was to use a strategy that if probability of doubling is high and value then go low, but does not work that well
    # for index, freq in enumerate(frequency):
    #     if p_doubling[index] > 0.5 and (sum([player.stripes for player in player_list]) > PLAYERCOUNT*cur_player.stripes):
    #         if sum(frequency) == DICE:
    #             cur_player.increment_times_one_first()
    #         return index, freq
    for index in range(len(frequency) - 1, -1, -1):
        freq = frequency[index]
        if freq > 0:
            give = 0
            receive = expected_stripes_for_self[index]
            # Calculate the score focusing on giving stripes
            score = -exponential_utility(receive+cur_player.stripes, cur_player.beta) * cur_player.get_own_utility()

            for player_index, player in enumerate(player_list):
                if player_index != cur_player.id - 1:
                    give += exponential_utility(p_doubling[index], player.alpha) * player.stripes * cur_player.utilities[player_index]
            if cur_player.id == len(player_list):
                give += expected_given_stripes_above_30[index] * cur_player.utilities[0]
            else:
                give += expected_given_stripes_above_30[index] * cur_player.utilities[cur_player.id]
            score += give


            if score > best_score:
                best_score = score
                best_index = index
                best_freq = freq

    if sum(frequency) == 6:
        for i in range(SIDES):
            if (frequency[i] > 0 and frequency[i] != 6) or (i == 0 and frequency[i] == 6):
                if i == best_index:
                    cur_player.increment_times_one_first()
                break
    return best_index, best_freq

def process_player_stripes(cur_sum: int, player_list: list[Player], cur_player: Player) -> None:
    """Function to process giving stripes to each player after a turn
    
    Args:
        cur_sum: the sum of the dice thrown by cur_player
        player_list: a list with all the players in the game
        cur_player: the current player
    """
    stripes_taken, stripes_given, doubled = calculate_rewards(cur_sum, player_list, cur_player)
    if stripes_given > 0 and not doubled:
        cur_player.add_stripes_given(stripes_given)
        cur_player.update_best_extra_above_30(stripes_given)
        player_list[cur_player.id % PLAYERCOUNT].add_stripes(stripes_given) 
    elif stripes_taken > 0:
        cur_player.add_stripes(stripes_taken)
    else:
        cur_player.add_stripes_given(stripes_given)
        cur_player.update_best_doubling(stripes_given)
        cur_player.increment_doubling_count()
        for player in player_list:
            if player.id == cur_player.id:
                continue
            player.double_stripes()
        


def calculate_rewards(dice_sum: int, player_list: list[Player], cur_player: Player) -> tuple[int, int, bool]:
    """Function used to calculate values in the format [stripes_taken, stripes_given, doubled],
    stripes_taken indicates number of stripes for yourself, stripes_given indicates how many stripes
    you gave to other players, doubled is only needed to know whether those stripes given is via doubling or not
    
    Args:
        cur_sum: the sum of the dice thrown by cur_player
        player_list: a list with all the players in the game
        cur_player: the current player
    """

    stripes_taken, stripes_given, doubled = 0, 0, False
    if dice_sum > 30:
        extra_stripes = dice_sum - 30
        stripes_given += extra_stripes
        dice_count = DICE
        while True:
            frequency = draw_dice(dice_count)
            if frequency[extra_stripes - 1] == 0:
                break
            dice_count -= frequency[extra_stripes - 1]
            stripes_given += extra_stripes * frequency[extra_stripes - 1]
            if dice_count == 0:
                dice_count = 6
    elif dice_sum <= 10:
        stripes_given = sum(player.stripes for player in player_list if player.id != cur_player.id)
        doubled = True
    else:
        stripes_taken = 30 - dice_sum
    return stripes_taken, stripes_given, doubled

def choices_QLearner(RL_agent: Agent):
    obs = draw_dice(DICE)
    cur_sum = 0
    reward = 0
    while any(val > 0 for val in obs):
        action = RL_agent.get_action(obs, cur_sum) # action is a number that they picked

        prev_sum = cur_sum
        dice_count, cur_sum = handle_dice(sum(obs), cur_sum, action, obs[action], RL_agent.id, dice_choice_freq_per_player)
        next_obs = draw_dice(dice_count)
        terminated = all(x == 0 for x in next_obs)
        if terminated:
            stripes_taken, stripes_given, doubled = calculate_rewards(cur_sum, player_list, player_list[RL_agent.id-1])
            reward = -stripes_taken
            if RL_agent.use_doubling or not doubled:
                reward += stripes_given
            
        RL_agent.update(prev_sum, obs, reward, terminated, next_obs)
        
        if terminated:
            break

        obs = next_obs

    RL_agent.decay()
    return cur_sum
       

    

def plot_frequencies(frequencies: list[int], title:str, xlabel:str, ylabel:str) -> None:
    plt.figure(figsize=(10, 6))
    keys = list(frequencies.keys())
    values = list(frequencies.values())
    sns.barplot(x=keys, y=values, hue=keys, palette='viridis', dodge=False, legend=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def play_round(player_list: list[Player], dice_choice_freq_per_player: list[dict[int,int]], rep: int):
    for id in range(1, PLAYERCOUNT + 1):
        cur_player = player_list[id - 1]
        strategy = cur_player.strategy
        
        if cur_player.stripes >= 15:
            cur_player.drink()
        dice_count = DICE
        cur_sum = 0
        temp = []
        
        if strategy != "QLearner" and strategy != "RiskTaker" and strategy != "RiskAverse" and strategy != "SmartRiskTaker":
            print([f"Player {player.strategy} has {player.stripes} stripes" for player in player_list])
        if (strategy == "QLearner"):
            cur_sum = choices_QLearner(cur_player.agent)
        else:
            while dice_count > 0:
                frequency = draw_dice(dice_count)
                if strategy == "RiskTaker":
                    temp = risk_strategy(frequency, dice_count, cur_sum, id, dice_choice_freq_per_player, player_list)
                elif strategy == "RiskAverse":
                    temp = risk_averse_strategy(frequency, dice_count, cur_sum, id, dice_choice_freq_per_player)
                elif strategy == "SmartRiskTaker":
                    expected_stripes_for_self, expected_given_stripes_above_30, p_doubling = prepare_smart_risktaker(frequency, dice_count, cur_sum)
                    best_index, best_freq = best_choice_smart_risktaker(frequency, expected_stripes_for_self, expected_given_stripes_above_30, p_doubling, player_list, id)
                    temp = handle_dice(dice_count, cur_sum, best_index, best_freq, id, dice_choice_freq_per_player)
                else:
                    temp = personal_strategy(frequency, dice_count, cur_sum, id, dice_choice_freq_per_player, player_list)
                dice_count = temp[0]
                cur_sum = temp[1]
        process_player_stripes(cur_sum, player_list, cur_player)
        if strategy != 'QLearner' or rep >= REPLICATIONS // 2:
            ending_sum_freq_per_player[id - 1][cur_sum] += 1  # Track ending sum for current player



def simulation(player_list: list[Player], dice_choice_freq_per_player: list[dict[int, int]]):
    for rep in tqdm(range(REPLICATIONS)):
        play_round(player_list, dice_choice_freq_per_player, rep)
    
    

def show_statistics(player_list: list[Player], dice_choice_freq_per_player: list[dict[int,int]], ending_sum_freq_per_player: list[dict[int,int]], rounds: int):
    for player in player_list:
        print("--------------------------------------------------------------------------------")
        print(player.strategy)
        print(f"player {player.id} had drunk {player.times_drunk / rounds:.3f} times per round and has given {player.stripes_given / rounds:.3f} stripes per round")
        print(f"player {player.id} went under 11 points {player.doubling_count / rounds:.4f} times per round")
        print(f"player {player.id} succeeded {player.get_doubling_success_rate():.4f} times with doubling")
        print(f"player {player.id}'s given/taken ratio is {player.get_given_taken_ratio():.4f}")
        print(f"player {player.id}'s best doubling amount was {player.best_doubling}")
        print(f"player {player.id}'s best extra stripes above 30 was {player.best_extra_above_30}")
    

        # Plot frequencies of dice chosen for the current player
        dice_freq_title = f"Frequency of Each Dice Chosen by Player {player.id}"
        plot_frequencies(
            dice_choice_freq_per_player[player.id - 1],
            dice_freq_title,
            "Dice Face",
            "Frequency"
        )

        # Plot frequencies of ending sums for the current player
        sum_freq_title = f"Frequency of Ending Sums of Dice for Player {player.id}"
        plot_frequencies(
            ending_sum_freq_per_player[player.id - 1],
            sum_freq_title,
            "Ending Sum",
            "Frequency"
        )
    print("--------------------------------------------------------------------------------")    





# actual game/simulation: 
player_list = []
for i in range(1, PLAYERCOUNT + 1):
    strategy = input(f"What is the strategy for player {i}? (RiskTaker, RiskAverse, SmartRiskTaker, QLearner or playSelf) ")
    if strategy == "playSelf":
        strategy = input("What is your name? ")
    alpha, beta = 1.1, 0.04
    if strategy == "QLearner":
        temp = input("Do you want to use rewards for doubling? (y/n) ")
        use_doubling = temp == "y"
        player_list.append(Player(i, strategy, alpha, beta, IS_SIMULATION, use_doubling))
    else:
        player_list.append(Player(i, strategy, alpha, beta, IS_SIMULATION))
for player in player_list:
    if (player.strategy == "SmartRiskTaker"):
        for index in range(len(player_list)):
            if player.id == (index+1):
                player.add_utility(PLAYERCOUNT-1)
            else:
                player.add_utility(1)


if IS_SIMULATION:
    simulation(player_list, dice_choice_freq_per_player)

SHOW_STATS = str(input("Do you want to view the statistics? (y/n) "))
if SHOW_STATS == 'y':   
    show_statistics(player_list, dice_choice_freq_per_player, ending_sum_freq_per_player, REPLICATIONS)

for player in player_list:
    if player.strategy == "QLearner":
        with open(f"q_table_{player.agent.use_doubling}.txt", "w") as f:
            for row in player.agent.q_values:
                f.write("  ".join(f"{val:.2f}" for val in row) + "\n")



exit()
    


def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

rolling_len = 500
fig, axs = plt.subplots(ncols=3, figsize=(12,5))

axs[0].set_title("Episode Rewards")
reward_moving_average = get_moving_avgs(
    env.return_queue,
    rolling_len,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)


axs[1].set_title("Episode Lengths")
length_moving_average = get_moving_avgs(
    env.length_queue,
    rolling_len,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)

axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(
    car.training_error,
    rolling_len,
    "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
plt.tight_layout()
plt.show()
 




# constants and initialization
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CHOOSE_COLOUR = (44, 117, 255)
HOLD_COLOUR = (139, 0, 0)
DICE_IMAGE_PATH = r"C:\Users\bartv\Documents\VScodeProjects\Python\Images"
FONT_PATH = r'C:\Users\bartv\Documents\VScodeProjects\Fonts\Roboto-Medium.ttf'
DICE_ICON_PATH = os.path.join(DICE_IMAGE_PATH, 'dice_icon.png')
ANIMATION_DURATION = 0.5  # Duration of the animation in seconds
WAIT_DURATION = 1.5 # Duration until next roll for AI turns

pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dertigen")
Icon = pygame.image.load(DICE_ICON_PATH)
pygame.display.set_icon(Icon)
clock = pygame.time.Clock()

class Dice:
    def __init__(self) -> None:
        self.value = rand.randint(1,6)
        self.chosen = False
        self.held = False
        self.image_path = DICE_IMAGE_PATH
        self.images = [pygame.transform.scale(pygame.image.load(os.path.join(self.image_path, f'dice_{i}.png')), (50, 50)) for i in range(1, 7)]
        self.animating = False
        self.animation_start_time = 0
    
    
    def roll(self) -> None:
        if not self.held and not self.chosen:
            self.value = rand.randint(1, 6)

    
    def start_animation(self):
        if not self.animating and not (self.held or self.chosen):
            self.animating = True
            self.animation_start_time = pygame.time.get_ticks()


    def update_animation(self):
        if self.animating:
            elapsed_time = (pygame.time.get_ticks() - self.animation_start_time) / 1000
            if elapsed_time < ANIMATION_DURATION:
                self.value = rand.randint(1, 6)
            else:
                self.animating = False
                self.roll()       

    def toggle_chosen(self):
        if not self.held:
            self.chosen = not self.chosen
       
    def draw(self, x, y):
        screen.blit(self.images[self.value - 1], (x, y))
        if self.chosen:
            pygame.draw.rect(screen, (CHOOSE_COLOUR), (x + 2, y + 2, 46, 46), 3)
        elif self.held:
            pygame.draw.rect(screen, (HOLD_COLOUR), (x + 2, y + 2, 46, 46), 3)

class Scoreboard:
    def __init__(self, x, y, width, height, players):
        self.rect = pygame.Rect(x, y, width, height)
        self.players = players
        self.column_names = players[0].get_columns()
        self.font = pygame.font.Font(FONT_PATH, 16)


    def draw(self):
        pygame.draw.rect(screen, (255, 223, 186), self.rect)  # Vibrant background colour
        pygame.draw.rect(screen, (255, 87, 34), self.rect, 5)  # Border colour and thickness

        # Calculate column width and header height
        column_width = self.rect.width // len(self.column_names)
        header_height = 30

        # Draw the column headers
        for i, column_name in enumerate(self.players[0].get_columns()):
            x = self.rect.x + i * column_width
            y = self.rect.y
            pygame.draw.rect(screen, (255, 87, 34), (x, y, column_width, header_height), 2)
            text = self.font.render(column_name, True, BLACK)
            screen.blit(text, (x + 5, y + 5))

        # Draw the player data rows
        row_height = 25
        for i, player in enumerate(self.players):
            for j, cell in enumerate(player.get_row()):
                x = self.rect.x + j * column_width
                y = self.rect.y + header_height + i * row_height
                pygame.draw.rect(screen, (255, 87, 34), (x, y, column_width, row_height), 1)
                text = self.font.render(str(cell), True, BLACK)
                screen.blit(text, (x + 5, y + 5))
        
       

class Slider:
    def __init__(self, x, y, width, height, name, start_value, min_val, max_val, step):
        self.rect = pygame.Rect(x, y, width, height)
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.value = start_value
        self.knob_rect = pygame.Rect(x, y, height, height)
        self.knob_colour = (0,150,255) # bright blue
        self.line_colour = (230,230,230) # light grey track
        self.fill_colour = (0,200,255) # bright cyan fill
        self.dragging = False
        self.update_knob_position()
        self.font = pygame.font.Font(FONT_PATH, 20)


    def update_knob_position(self):
        rel_x = (self.value - self.min_val) / (self.max_val - self.min_val) * (self.rect.width - self.knob_rect.width)
        self.knob_rect.x = self.rect.x + rel_x

    def draw(self, screen):
         # Draw the filled line
        fill_width = self.knob_rect.centerx - self.rect.x
        pygame.draw.line(screen, self.fill_colour, (self.rect.x, self.rect.centery), (self.rect.x + fill_width, self.rect.centery), 4)
        # Draw the line
        pygame.draw.line(screen, self.line_colour, (self.rect.x + fill_width, self.rect.centery), (self.rect.right, self.rect.centery), 4)
        # Draw the knob with border
        pygame.draw.ellipse(screen, self.knob_colour, self.knob_rect)
        # Draw the current value above the knob
        value_text = f"{self.value:.1f}"
        text_surf = self.font.render(value_text, True, self.knob_colour)
        screen.blit(text_surf, (self.knob_rect.x + self.knob_rect.width // 2 - text_surf.get_width() // 2, self.knob_rect.y - 30))
        # Draw the name of the slider above the line
        name_text = self.name
        name_surf = self.font.render(name_text, True, self.line_colour)
        screen.blit(name_surf, (self.rect.x + self.rect.width // 2 - name_surf.get_width() // 2, self.rect.y - 55))
        

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.knob_rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self.knob_rect.x = max(self.rect.x, min(event.pos[0] - self.knob_rect.width // 2, self.rect.right - self.knob_rect.width))
                self.update_value()

    def update_value(self):
        rel_x = self.knob_rect.x - self.rect.x
        self.value = round((rel_x / (self.rect.width - self.knob_rect.width)) * (self.max_val - self.min_val) / self.step) * self.step + self.min_val
        self.value = max(self.min_val, min(self.value, self.max_val))

    def get_value(self):
        return self.value


class Button:
    def __init__(self, x, y, width, height, text, action, enabled, font):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action = action
        self.enabled = enabled
        self.font = font
        self.hovered = False
        self.normal_colour = WHITE
        self.hover_colour = pygame.Color(180, 180, 180)
        self.disabled_colour = pygame.Color(200, 200, 200)
        


    def draw(self):
        colour = self.normal_colour if self.enabled else self.disabled_colour
        if self.hovered and self.enabled:
            colour = self.hover_colour
        pygame.draw.rect(screen, colour, self.rect, 0, 3)
        text_surf = self.font.render(self.text, True, BLACK)
        screen.blit(text_surf, text_surf.get_rect(center=self.rect.center))

    def click(self, event):
        if self.rect.collidepoint(event.pos) and self.enabled:
            self.action()
    
    def set_enabled(self, enabled):
        self.enabled = enabled

    def handle_hover(self,event):
        self.hovered = self.rect.collidepoint(event.pos)


class UIManager:
    def __init__(self, dice, buttons, sliders, popup_message, game):
        self.dice = dice
        self.buttons = buttons
        self.sliders = sliders
        self.popup_message = popup_message
        self.game = game

    def draw(self):
        for i, die in enumerate(self.dice):
            die.update_animation()
            die.draw(100 + i * 60, 200)

        for button in self.buttons:
            if button.text == "End Turn":
                button.set_enabled(game.all_dice_held_or_chosen())
            elif button.text == "Drink":
                button.set_enabled(self.game.players[self.game.current_player_index].stripes >= 15)
            elif button.text == "Roll":
                button.set_enabled(any([die.chosen for die in self.dice]))
                if game.all_dice_held_or_chosen():
                    button.set_enabled(False)
            button.draw()

        for slider in self.sliders:
            slider.draw(screen)

        if self.popup_message:
            self.draw_popup()

    
    def draw_popup(self):
        popup_rect = pygame.Rect(WIDTH // 4, HEIGHT // 4, WIDTH // 2, HEIGHT // 2)
        pygame.draw.rect(screen, (255, 223, 186), popup_rect)  # Vibrant background colour
        pygame.draw.rect(screen, (255, 87, 34), popup_rect, 5)  # Border colour and thickness

        text_name = self.game.font.render(self.game.players[self.game.current_player_index].strategy, True, (0,0,0))
        text_name_rect = text_name.get_rect(center=popup_rect.center)
        text_name_rect.y -= 20  # Move down by 20 pixels
        screen.blit(text_name, text_name_rect)

        text_surf = self.game.font.render(self.popup_message, True, (0, 0, 0))
        text_surf_rect = text_surf.get_rect(center=popup_rect.center)
        text_surf_rect.y += 20  # Move up by 20 pixels
        screen.blit(text_surf, text_surf_rect)
         

class Game:
    def __init__(self, playerList):
        self.dice = [Dice() for _ in range(6)]
        self.players = playerList
        self.current_player_index = 0
        self.is_special_turn = False
        self.extra_stripes = 0
        self.stripes = 0
        self.font = pygame.font.Font(FONT_PATH, 30)
        self.rounds = 0
        self.buttons = [
            Button(100, 500, 100, 50, "Roll", self.start_roll_animation, False, self.font),
            Button(300, 500, 120, 50, "End Turn", self.end_turn, False, self.font),
            Button(500, 500, 100, 50, "Drink", self.take_drink, False, self.font)
        ]
        self.sliders = [
            Slider(600, 300, 100, 30, "Rolling Time", ANIMATION_DURATION, 0, 3, 0.1),
            Slider(600, 400, 100, 30, "Wait Time" ,WAIT_DURATION, 0, 5, 0.2)
        ]
        self.ui_manager = UIManager(self.dice, self.buttons, self.sliders, None, self)
        self.scoreboard = Scoreboard(WIDTH // 10, HEIGHT // 10, 4 * WIDTH // 5, 4 * HEIGHT // 5, self.players)
        self.special_turn_next_step_time = 0
        self.animation_starting_time = 0
        self.last_action_time = None  # Initialize last_action_time to None
        self.last_display_time = None
        self.dice_display_start_time = None

        
    def lock_die(self, die):
        if die.chosen:
            die.chosen = False
            die.held = True

    # need to make sure this is only called once!!!
    def start_roll_animation(self):
        self.animation_starting_time = pygame.time.get_ticks()
        for die in self.dice:
            self.lock_die(die)
            die.start_animation()

    def end_turn(self): 
        cur_player = self.players[self.current_player_index]
        cur_sum = sum(die.value for die in self.dice)
        ending_sum_freq_per_player[self.current_player_index][cur_sum]+=1

        if cur_sum <= 30:
            process_player_stripes(cur_sum, self.players, cur_player)
            self.next_player()
        else:
            self.reset_turn()
            
            self.extra_stripes = cur_sum - 30
            self.stripes = self.extra_stripes

            self.is_special_turn = True
            self.special_turn_next_step_time = pygame.time.get_ticks() + 1000 * WAIT_DURATION  # 1.5 second delay
        self.start_roll_animation()


    def take_drink(self):
        drinks = self.players[self.current_player_index].drink() / 2
        drinks_word = self.convert_to_words(drinks)
        if drinks <= 1:
            self.ui_manager.popup_message = f"You need to take {drinks_word} drink"
        else:
            self.ui_manager.popup_message = f"You need to take {drinks_word} drinks"
        

    def convert_to_words(self, value):
        # Define the mapping for drink values
        drink_words = {
            0.5: "half a",
            1: "one",
            1.5: "one and a half",
            2: "two",
            2.5: "two and a half",
            3: "three",
            3.5: "three and a half",
            4: "four",
            4.5: "four and a half",
            5: "five"
        }
        # Return the corresponding word for the value or a default message
        return drink_words.get(value, "an enormous number of")

    def reset_turn(self):
        for die in self.dice:
            die.held = False
            die.chosen = False
            

    def roll_dice(self):
        for die in self.dice:
            die.roll()

    def next_player(self):
        if self.current_player_index == 0:
            self.rounds +=1
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        self.reset_turn()

    def draw(self):
        screen.fill(BLACK)
        current_player = self.players[self.current_player_index]
        player_text = self.font.render(f"Current Player: {current_player.strategy}", True, WHITE)
        screen.blit(player_text, (100, 50))
        
        for i, player in enumerate(self.players):
            player.update_stripes()
            player_stripes_text = self.font.render(f"{player.strategy}: {player.stripes}", True, WHITE)
            screen.blit(player_stripes_text, (WIDTH - 250, 10 + i * 30))

        self.ui_manager.draw()
        
        

    def choose_all_dice(self, value):
        for die in self.dice:
            if not die.held:
                die.chosen = False
        for die in self.dice:
            if die.value == value and not die.held:
                die.toggle_chosen()
                

    def all_dice_held_or_chosen(self):
        return all(die.held or die.chosen for die in self.dice)
    
    def get_choice_computer(self, cur_player):
            choice = 0
            if cur_player.strategy == "RiskTaker":
                choice = min([die.value for die in self.dice if not (die.held or die.chosen)])
            elif cur_player.strategy == "RiskAverse":
                choice = max([die.value for die in self.dice if not (die.held or die.chosen)])
            else:
                frequency = [0] * 6
                diceCount = 6
                curSum = 0
                for die in self.dice:
                    if not (die.held or die.chosen):
                        frequency[die.value-1]+=1
                    else:
                        diceCount -=1
                        curSum += die.value
                expectedStripesForSelf, expectedGivenStripesAbove30, p_doubling = prepare_smart_risktaker(frequency, diceCount, curSum)
                choice, _ = best_choice_smart_risktaker(frequency, expectedStripesForSelf, expectedGivenStripesAbove30, p_doubling, self.players, cur_player.id)
                choice += 1 # in the method we get an value which is 0-indexed
            return choice


    def computer_turn(self):
        cur_player = self.players[self.current_player_index]
        if cur_player.pending_stripes == 0:
            cur_player.drink() 
        current_time = pygame.time.get_ticks()

        # Check if it's time for the next step in the special turn
        if current_time >= self.special_turn_next_step_time:
            # Ensure all dice animations have stopped
            if not any(die.animating for die in self.dice):
                # Initialize last_action_time if not already set
                if self.last_action_time is None:
                    self.last_action_time = current_time

                # If enough time has passed since the last action, proceed
                if current_time - self.last_action_time >= int(WAIT_DURATION * 500):
                    choice = self.get_choice_computer(cur_player)
                    self.choose_all_dice(choice)
                    self.special_turn_next_step_time = current_time + WAIT_DURATION * 1000
                    self.start_roll_animation()

                    # Update the last action time
                    self.last_action_time = None

        if self.all_dice_held_or_chosen():
            self.end_turn()
        

    def handle_special_turn(self):
        current_time = pygame.time.get_ticks()

        if current_time >= self.special_turn_next_step_time:  # Check if enough time has passed
            if not any(die.value == self.extra_stripes for die in self.dice if not (die.held or die.chosen)):
                # No remaining dice to choose, finish the turn
                for die in self.dice:
                    if die.held:
                        self.stripes += self.extra_stripes

                self.is_special_turn = False
                self.players[self.current_player_index].add_stripes_given(self.stripes)
                self.players[self.current_player_index].update_bestExtraAbove30(self.stripes)
                self.players[(self.current_player_index + 1) % len(self.players)].add_stripes(self.stripes)
                self.next_player()
                self.start_roll_animation()
                return

       
            

            # Initialize dice_display_start_time if it hasn't been set
            if self.dice_display_start_time is None:
                self.dice_display_start_time = current_time  # Start the timer for displaying the dice

            # Check if the dice have been displayed long enough
            if current_time - self.dice_display_start_time >= int(WAIT_DURATION * 500):
                self.choose_all_dice(self.extra_stripes)
                self.special_turn_next_step_time = current_time + WAIT_DURATION * 1000  # 1.5 seconds delay for next turn
                self.start_roll_animation()
                self.dice_display_start_time = None  # Reset the display timer

                if self.all_dice_held_or_chosen():
                    self.stripes += self.extra_stripes * DICE
                

# Game loop flag
running = True
# value for the scoreboard
tab_key_pressed = False
# Initialize the game with players
game = Game(player_list)
AI = ["RiskTaker", "RiskAverse", "SmartRiskTaker"]

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if game.ui_manager.popup_message:
                game.ui_manager.popup_message = None
            else:
                for button in game.buttons:
                    button.click(event)
                x, y = event.pos
                for i, die in enumerate(game.dice):
                    if 100 + i * 60 < x < 150 + i * 60 and 200 < y < 250:
                        game.choose_all_dice(die.value)
        elif event.type == pygame.MOUSEMOTION:
            for button in game.buttons:
                button.handle_hover(event)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_TAB:
                tab_key_pressed = True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_TAB:
                tab_key_pressed = False

         # Handle events for sliders
        for slider in game.sliders:
            slider.handle_event(event)
        ANIMATION_DURATION = game.sliders[0].get_value()
        WAIT_DURATION = game.sliders[1].get_value()    

    if game.is_special_turn:
        game.handle_special_turn()
    elif game.players[game.current_player_index].strategy in AI:
        game.computer_turn()
    
    game.draw()
    if tab_key_pressed:
        game.scoreboard.draw()

    pygame.display.flip()
    clock.tick(30)
    
pygame.quit()

if input("Do you want to view the statistics? ").strip().lower() == "yes":
    show_statistics(player_list, dice_choice_freq_per_player, ending_sum_freq_per_player, game.rounds)





# to make sure I can do the animation of adding stripes, I will need a list that holds the info on who gets extra stripes
# further I need two pygame.time.get_ticks() 1 is for the timer of the f"{sign}{stripes}" text and the other will be the animation
# that enumerates the new stripes to the player, so if you have 0 stripes and now got 4, every 0.05 seconds you will gain 1 stripe, up until your new count