import random 

def selection(population):

    index_1 = random.randint(0,len(population)-1)
    index_2 = random.randint(0,len(population)-1)
    index_3 = random.randint(0,len(population)-1)

    while(index_1 == index_2 or index_1 == index_3 or index_2 == index_3):
        index_2 = random.randint(0,len(population)-1)
        index_3 = random.randint(0,len(population)-1)

    individual_1 = population[index_1]
    individual_2 = population[index_2]
    individual_3 = population[index_3]

    if(individual_1.fitness >= individual_2.fitness and individual_1.fitness >= individual_3.fitness):
        return individual_1
    elif(individual_2.fitness >= individual_1.fitness and individual_2.fitness >= individual_3.fitness):
        return individual_2
    else:
        return individual_3