"""
__author__      = "Jithin Pradeep"
__version__     = "1.0"
__email__       = "jithinpr2@gmail.com"

An Implementation of Genetic Algorithm in python
"""
import math
import random
import sys

START_SIZE = 75  # Population size at start.
MAX_EPOCHS = 1000  # Arbitrary number of test cycles.
MATING_PROBABILITY = 0.7  # Probability of two chromosomes mating. Range: 0.0 < MATING_PROBABILITY < 1.0
MUTATION_RATE = 0.001  # Mutation Rate. Range: 0.0 < MUTATION_RATE < 1.0
MIN_SELECT = 10  # Minimum parents allowed for selection.
MAX_SELECT = 50  # Maximum parents allowed for selection. Range: MIN_SELECT < MAX_SELECT < START_SIZE
OFFSPRING_PER_GENERATION = 20  # New offspring created per generation. Range: 0 < OFFSPRING_PER_GENERATION < MAX_SELECT.
MINIMUM_SHUFFLES = 8  # For randomizing starting chromosomes
MAXIMUM_SHUFFLES = 20

SHOW_VERBOSE_RESULTS = False

CITY_COUNT = 8
TARGET = 86.6299  # Number for algorithm to find.
XLocs = [30, 40, 40, 29, 19, 9, 9, 20]
YLocs = [5, 10, 20, 25, 25, 19, 9, 5]


class TSP1:
    def __init__(self, startSize, epochs, matingProb, mutationRate, minSelect, maxSelect, generation, minShuffles,
                 maxShuffles, verbose, cityCount, target, xlocs, ylocs):
        self.mStartSize = startSize
        self.mEpochs = epochs
        self.mMatingProbability = matingProb
        self.mMutationRate = mutationRate
        self.mMinSelect = minSelect
        self.mMaxSelect = maxSelect
        self.mOffspringPerGeneration = generation
        self.mMinimumShuffles = minShuffles
        self.mMaximumShuffles = maxShuffles
        self.mShowVerboseResults = verbose
        self.mCityCount = cityCount
        self.mTarget = target
        self.mXCoords = xlocs
        self.mYCoords = ylocs

        self.population = []
        self.cityMap = []
        self.epoch = 0
        self.childCount = 0
        self.nextMutation = 0  # For scheduling mutations.
        self.mutations = 0
        return

    def initialize_city_map(self):
        for i in range(self.mCityCount):
            newCity = City()
            newCity.set_x(self.mXCoords[i])
            newCity.set_y(self.mYCoords[i])
            self.cityMap.append(newCity)

        return

    def get_exclusive_random_integer(self, high, numberA):
        done = False
        numberB = 0

        while not done:
            numberB = random.randrange(0, high)
            if numberB != numberA:
                done = True

        return numberB

    def exchange_mutation(self, index, exchanges):
        i = 0
        thisChromo = Chromosome(self.mCityCount)
        done = False

        thisChromo = self.population[index]

        while not done:
            gene1 = random.randrange(0, self.mCityCount)
            gene2 = self.get_exclusive_random_integer(self.mCityCount, gene1)

            # Exchange the chosen genes.
            tempData = thisChromo.get_data(gene1)
            thisChromo.set_data(gene1, thisChromo.get_data(gene2))
            thisChromo.set_data(gene2, tempData)

            if i == exchanges:
                done = True

            i += 1

        self.mutations += 1
        return

    def get_distance(self, firstCity, secondCity):
        cityA = City()
        cityB = City()
        cityA = self.cityMap[firstCity]
        cityB = self.cityMap[secondCity]
        a2 = math.pow(math.fabs(cityA.get_x() - cityB.get_x()), 2)
        b2 = math.pow(math.fabs(cityA.get_y() - cityB.get_y()), 2)
        return math.sqrt(a2 + b2)

    def get_total_distance(self, chromoIndex):
        thisChromo = Chromosome(self.mCityCount)
        thisChromo = self.population[chromoIndex]

        for i in range(self.mCityCount):
            if i == self.mCityCount - 1:
                thisChromo.set_total(
                    thisChromo.get_total() + self.get_distance(thisChromo.get_data(self.mCityCount - 1),
                                                               thisChromo.get_data(0)))  # Complete trip.
            else:
                thisChromo.set_total(
                    thisChromo.get_total() + self.get_distance(thisChromo.get_data(i), thisChromo.get_data(i + 1)))

        return

    def initialize_chromosomes(self):
        shuffles = 0
        chromoIndex = 0

        for i in range(self.mStartSize):
            newChromo = Chromosome(self.mCityCount)
            for j in range(self.mCityCount):
                newChromo.set_data(j, j)

            self.population.append(newChromo)
            chromoIndex = len(self.population) - 1

            # Randomly choose the number of shuffles to perform.
            shuffles = random.randrange(self.mMinimumShuffles, self.mMaximumShuffles)

            self.exchange_mutation(chromoIndex, shuffles)

            self.get_total_distance(chromoIndex)

        return

    def math_round(self, inValue):
        outValue = 0
        if math.modf(inValue)[0] >= 0.5:
            outValue = math.ceil(inValue)
        else:
            outValue = math.floor(inValue)
        return outValue

    def get_maximum(self):
        # Returns an array index.
        popSize = 0;
        thisChromo = Chromosome(self.mCityCount)
        thatChromo = Chromosome(self.mCityCount)
        maximum = 0
        foundNewMaximum = False
        done = False

        while not done:
            foundNewMaximum = False
            popSize = len(self.population)
            for i in range(popSize):
                if i != maximum:
                    thisChromo = self.population[i]
                    thatChromo = self.population[maximum]
                    # The maximum has to be in relation to the Target.
                    if math.fabs(self.mTarget - thisChromo.get_total()) > math.fabs(
                                    self.mTarget - thatChromo.get_total()):
                        maximum = i
                        foundNewMaximum = True

            if foundNewMaximum == False:
                done = True

        return maximum

    def get_minimum(self):
        # Returns an array index.
        popSize = 0;
        thisChromo = Chromosome(self.mCityCount)
        thatChromo = Chromosome(self.mCityCount)
        minimum = 0
        foundNewMinimum = False
        done = False

        while not done:
            foundNewMinimum = False
            popSize = len(self.population)
            for i in range(popSize):
                if i != minimum:
                    thisChromo = self.population[i]
                    thatChromo = self.population[minimum]
                    # The minimum has to be in relation to the Target.
                    if math.fabs(self.mTarget - thisChromo.get_total()) < math.fabs(
                                    self.mTarget - thatChromo.get_total()):
                        minimum = i
                        foundNewMinimum = True

            if foundNewMinimum == False:
                done = True

        return minimum

    def get_fitness(self):
        # Lowest errors = 100%, Highest errors = 0%
        popSize = 0
        thisChromo = Chromosome(self.mCityCount)
        bestScore = 0.0
        worstScore = 0.0

        # The worst score would be the one furthest from the Target.
        thisChromo = self.population[self.get_maximum()]
        worstScore = math.fabs(self.mTarget - thisChromo.get_total())

        # Convert to a weighted percentage.
        thisChromo = self.population[self.get_minimum()]
        bestScore = worstScore - math.fabs(self.mTarget - thisChromo.get_total())

        popSize = len(self.population)
        for i in range(popSize):
            thisChromo = self.population[i]
            thisChromo.set_fitness((worstScore - (math.fabs(self.mTarget - thisChromo.get_total()))) * 100 / bestScore)

        return

    def roulette_selection(self):
        j = 0
        popSize = 0
        genTotal = 0.0
        selTotal = 0.0
        rouletteSpin = 0.0
        thisChromo = Chromosome(self.mCityCount)
        thatChromo = Chromosome(self.mCityCount)
        done = False

        popSize = len(self.population)
        for i in range(popSize):
            thisChromo = self.population[i]
            genTotal += thisChromo.get_fitness()

        genTotal *= 0.01

        for i in range(popSize):
            thisChromo = self.population[i]
            thisChromo.set_selection_probability(thisChromo.get_fitness() / genTotal)

        for i in range(self.mOffspringPerGeneration):
            rouletteSpin = random.randrange(0, 99)
            j = 0
            selTotal = 0
            done = False
            while not done:
                thisChromo = self.population[j]
                selTotal += thisChromo.get_selection_probability()
                if selTotal >= rouletteSpin:
                    if j == 0:
                        thatChromo = self.population[j]
                    elif j >= popSize - 1:
                        thatChromo = self.population[popSize - 1]
                    else:
                        thatChromo = self.population[j - 1]

                    thatChromo.set_selected(True)
                    done = True
                else:
                    j += 1

        return

    def choose_first_parent(self):
        parent = 0
        thisChromo = Chromosome(self.mCityCount)
        done = False

        while not done:
            # Randomly choose an eligible parent.
            parent = random.randrange(0, len(self.population) - 1)
            thisChromo = self.population[parent]
            if thisChromo.get_selected() == True:
                done = True

        return parent

    def choose_second_parent(self, parentA):
        parentB = 0
        thisChromo = Chromosome(self.mCityCount)
        done = False

        while not done:
            # Randomly choose an eligible parent.
            parentB = random.randrange(0, len(self.population) - 1)
            if parentB != parentA:
                thisChromo = self.population[parentB]
                if thisChromo.get_selected() == True:
                    done = True

        return parentB

    def partially_mapped_crossover(self, chromA, chromB, childIndex):
        j = 0
        item1 = 0
        item2 = 0
        pos1 = 0
        pos2 = 0
        thisChromo = Chromosome(self.mCityCount)
        thisChromo = self.population[chromA]
        thatChromo = Chromosome(self.mCityCount)
        thatChromo = self.population[chromB]
        newChromo = Chromosome(self.mCityCount)
        newChromo = self.population[childIndex]

        crossPoint1 = random.randrange(self.mCityCount);
        crossPoint2 = self.get_exclusive_random_integer(self.mCityCount, crossPoint1)
        if crossPoint2 < crossPoint1:
            j = crossPoint1
            crossPoint1 = crossPoint2
            crossPoint2 = j

        # Copy parentA genes to offspring.
        for i in range(self.mCityCount):
            newChromo.set_data(i, thisChromo.get_data(i))

        for i in range(crossPoint1, crossPoint2 + 1):
            # Get the two items to swap.
            item1 = thisChromo.get_data(i)
            item2 = thatChromo.get_data(i)

            # Get the items' positions in the offspring.
            for k in range(self.mCityCount):
                if newChromo.get_data(k) == item1:
                    pos1 = k
                elif newChromo.get_data(k) == item2:
                    pos2 = k

            # Swap them.
            if item1 != item2:
                newChromo.set_data(pos1, item2)
                newChromo.set_data(pos2, item1)

        return

    def do_mating(self):
        getRand = 0
        parentA = 0
        parentB = 0
        newChildIndex = 0

        for i in range(self.mOffspringPerGeneration):
            parentA = self.choose_first_parent()
            # Test probability of mating.
            getRand = random.randrange(0, 100)
            if getRand <= self.mMatingProbability * 100:
                parentB = self.choose_second_parent(parentA)
                newChromo = Chromosome(self.mCityCount)
                self.population.append(newChromo)
                newChildIndex = len(self.population) - 1
                self.partially_mapped_crossover(parentA, parentB, newChildIndex)
                if self.childCount == self.nextMutation:
                    getRand = random.randrange(self.mCityCount)
                    self.exchange_mutation(newChildIndex, 1)

                self.get_total_distance(newChildIndex)

                self.childCount += 1

                # Schedule next mutation.
                if math.fmod(self.childCount, self.math_round(1.0 / MUTATION_RATE)) == 0:
                    self.nextMutation = self.childCount + random.randrange(self.math_round(1.0 / self.mMutationRate))

        return

    def prep_next_epoch(self):
        popSize = 0;
        thisChromo = Chromosome(self.mCityCount)

        # Reset flags for selected individuals.
        popSize = len(self.population)
        for i in range(popSize):
            thisChromo = self.population[i]
            thisChromo.set_selected(False)

        return

    def print_best_from_population(self):
        popSize = 0
        tempString = ""
        thisChromo = Chromosome(self.mCityCount)

        popSize = len(self.population)
        for i in range(popSize):
            thisChromo = self.population[i]
            if thisChromo.get_fitness() > 80.0:
                for j in range(self.mCityCount):
                    tempString += str(thisChromo.get_data(j))

                tempString += "; Dist = " + str(thisChromo.get_total()) + "; at " + str(
                    self.math_round(thisChromo.get_fitness())) + "%\n"

        return tempString

    def genetic_algorithm(self):
        popSize = 0
        thisChromo = Chromosome(self.mCityCount)
        done = False

        self.mutations = 0
        self.nextMutation = random.randrange(self.math_round(1.0 / self.mMutationRate))

        while not done:
            popSize = len(self.population)
            for i in range(popSize):
                thisChromo = self.population[i]
                if math.fabs(thisChromo.get_total() - self.mTarget) <= 1.0 or self.epoch == self.mEpochs:
                    done = True

            self.get_fitness()

            if self.mShowVerboseResults == True:
                popSize = len(self.population)
                for i in range(popSize):
                    thisChromo = self.population[i]
                    for j in range(self.mCityCount):
                        sys.stdout.write(str(thisChromo.get_data(j)))

                    sys.stdout.write(" = " + str(thisChromo.get_total()))
                    sys.stdout.write("\t" + str(thisChromo.get_fitness()) + "%\n")

                sys.stdout.write("\n")

            self.roulette_selection()

            self.do_mating()

            self.prep_next_epoch()

            self.epoch += 1

            sys.stdout.write("Epoch: " + str(self.epoch) + "\n")

        sys.stdout.write(self.print_best_from_population() + "\n")
        sys.stdout.write("done.\n")

        if self.epoch != self.mEpochs:
            popSize = len(self.population)
            for i in range(popSize):
                thisChromo = self.population[i]
                if math.fabs(thisChromo.get_total() - self.mTarget) <= 0.001:
                    # Print the chromosome.
                    for j in range(self.mCityCount):
                        sys.stdout.write(str(thisChromo.get_data(j)) + ", ")

                    sys.stdout.write("\n")

        sys.stdout.write("Completed " + str(self.epoch) + " epochs.\n")
        sys.stdout.write(
            "Encountered " + str(self.mutations) + " mutations in " + str(self.childCount) + " offspring.\n")

        return


class Chromosome:
    def __init__(self, cityCount):
        self.mData = [0] * cityCount
        self.mRegion = 0
        self.mTotal = 0
        self.mFitness = 0.0
        self.mSelected = False
        self.mAge = 0
        self.mSelectionProbability = 0.0
        return

    def set_selection_probability(self, probability):
        self.mSelectionProbability = probability
        return

    def get_selection_probability(self):
        return self.mSelectionProbability

    def set_age(self, epoch):
        self.mAge = epoch
        return

    def get_age(self):
        return self.mAge

    def set_selected(self, isSelected):
        self.mSelected = isSelected
        return

    def get_selected(self):
        return self.mSelected

    def set_fitness(self, score):
        self.mFitness = score
        return

    def get_fitness(self):
        return self.mFitness

    def set_total(self, value):
        self.mTotal = value
        return

    def get_total(self):
        return self.mTotal

    def set_region(self, region):
        self.mRegion = region
        return

    def get_region(self):
        return self.mRegion

    def set_data(self, index, value):
        self.mData[index] = value
        return

    def get_data(self, index):
        return self.mData[index]


class City:
    def __init__(self):
        self.mX = 0;
        self.mY = 0
        return

    def set_x(self, xCoordinate):
        self.mX = xCoordinate
        return

    def get_x(self):
        return self.mX

    def set_y(self, yCoordinate):
        self.mY = yCoordinate
        return

    def get_y(self):
        return self.mY


if __name__ == '__main__':
    tsp1 = TSP1(START_SIZE, MAX_EPOCHS, MATING_PROBABILITY, MUTATION_RATE, MIN_SELECT, MAX_SELECT,
                OFFSPRING_PER_GENERATION, MINIMUM_SHUFFLES, MAXIMUM_SHUFFLES, SHOW_VERBOSE_RESULTS, CITY_COUNT, TARGET,
                XLocs, YLocs)
    tsp1.initialize_city_map()
    tsp1.initialize_chromosomes()
    tsp1.genetic_algorithm()

