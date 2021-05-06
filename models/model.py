import numpy
import pygad
import pygad.nn
import pygad.gann
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('symptoms.csv')

# creating instance of labelencoder
labelencoder = LabelEncoder()

# Assigning numerical values and storing in another column
df['type'] = labelencoder.fit_transform(df['TYPE'])
    
#Droping Type column
df.drop('TYPE', axis='columns', inplace=True)

#Splitting the data set
data_input = df.loc[:,df.columns!='type']
data_output = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(data_input ,data_output,test_size=.42)

#converting to numpy array
data_inputs = X_train.to_numpy()
data_outputs = y_train.to_numpy()

class ModelAnn:

    def fitness_func(solution, sol_idx):
        global GANN_instance, data_inputs, data_outputs
        
        predictions = pygad.nn.predict(last_layer=GANN_instance.population_networks[sol_idx],
                                       data_inputs=data_inputs)
        
        correct_predictions = numpy.where(predictions == data_outputs)[0].size
        solution_fitness = (correct_predictions/data_outputs.size)*100
        
        return solution_fitness
    
    def callback_generation(ga_instance):
        global GANN_instance
        
        population_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks, 
                                                                population_vectors=ga_instance.population)
        
        GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)
        
        print("Generation = {generation}".format(generation=ga_instance.generations_completed))
        print("Accuracy   = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    
    def predict(inputs, solution_idx):
        prediction = pygad.nn.predict(last_layer=GANN_instance.population_networks[solution_idx],
                               data_inputs=inputs)
        return prediction

GANN_instance = pygad.gann.GANN(num_solutions=10,
                                num_neurons_input=20,
                                num_neurons_hidden_layers=[2],
                                num_neurons_output=4,
                                hidden_activations=["relu"],
                                output_activation="softmax")

population_vectors = pygad.gann.population_as_vectors(population_networks=GANN_instance.population_networks)

ga_instance = pygad.GA(num_generations=50, 
                       num_parents_mating=3, 
                       initial_population=population_vectors.copy(),
                       fitness_func=ModelAnn.fitness_func,
                       mutation_percent_genes=5,
                       callback_generation=ModelAnn.callback_generation)

ga_instance.run()


ga_instance.plot_result()

global solution_idx 
solution, solution_fitness, solution_idx = ga_instance.best_solution()

elast_layer = GANN_instance.population_networks[solution_idx]

print(solution)
print(solution_fitness)
print(solution_idx)

pickle.dump(elast_layer, open('GA.pkl', 'wb'))




