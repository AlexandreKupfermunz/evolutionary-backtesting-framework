import random

THRESHOLD_RATIO_STEP = 0.25

def mutation(individual):

    random_number = random.randint(0,149)
    
    if random_number < 5:

        difference = random.randint(1,2)
        sign = random_sign()

        new_value = individual.min_impulse_candles+sign*difference

        if new_value < 2:
            new_value = 2
        elif new_value > 15:
            new_value = 15

        if new_value <  individual.min_imbalance_count:
             individual.min_imbalance_count = new_value

        individual.min_impulse_candles = new_value

    elif random_number < 10:

        difference_in_percent = random.randint(1,10)/100
        sign = random_sign()

        new_value = individual.max_duration_ms + individual.max_duration_ms*sign*difference_in_percent

        if new_value < 50:
            new_value = 50
        elif new_value > 4000:
            new_value = 4000

        individual.max_duration_ms = int(new_value)

    elif random_number < 15:

        sign = random_sign()

        new_value = individual.diagonal_imbalance_ratio_threshold + sign*THRESHOLD_RATIO_STEP

        if new_value < 3:
            new_value = 3
        elif new_value > 10:
            new_value = 10
    
        individual.diagonal_imbalance_ratio_threshold = round(new_value, 2)

    elif random_number < 20:

        difference = 1
        sign = random_sign()

        new_value = individual.min_imbalance_count + sign*difference

        if new_value < 1:
            new_value = 1
        elif new_value > individual.min_impulse_candles:
            new_value = individual.min_impulse_candles
    
        individual.min_imbalance_count = new_value

    elif random_number < 25:

        difference = random.randint(1,2)
        sign = random_sign()

        new_value = individual.take_profit_ticks + sign*difference

        if new_value < 2:
            new_value = 2
        elif new_value > 30:
            new_value = 30
    
        individual.take_profit_ticks = new_value

    elif random_number < 30:

        difference = random.randint(1,2)
        sign = random_sign()

        new_value = individual.stop_loss_ticks + sign*difference

        if new_value < 2:
            new_value = 2
        elif new_value > 30:
            new_value = 30
    
        individual.stop_loss_ticks = new_value
 
    return individual

def random_sign():

    return random.choice([-1, 1])