import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Create the fuzzy logic variables and their linguistic terms
# For input variables
dirt_level = ctrl.Antecedent(np.arange(0, 11, 1), 'Dirt_Level')  # scale from 1 - 10
load_size = ctrl.Antecedent(np.arange(0, 101, 20), 'Load_Size')  # percent

# For output variable
washing_time = ctrl.Consequent(np.arange(0, 5, 1), 'Washing_Time')

# Membership functions for dirt level variable (Fuzzification)
dirt_level['Lightly_Soiled'] = fuzz.trapmf(dirt_level.universe, [0, 0, 2, 3])
dirt_level['Moderately_Soiled'] = fuzz.trapmf(dirt_level.universe, [2, 4, 6, 8])
dirt_level['Heavily_Soiled'] = fuzz.trapmf(dirt_level.universe, [6, 8, 10, 10])

# Membership functions for load size variable (Fuzzification)
load_size['Small'] = fuzz.trapmf(load_size.universe, [0, 0, 40, 60])
load_size['Medium'] = fuzz.trapmf(load_size.universe, [40, 60, 60, 80])
load_size['Large'] = fuzz.trapmf(load_size.universe, [60, 80, 100, 100])

# Membership functions for washing time variable (Fuzzification)
washing_time['Short'] = fuzz.trapmf(washing_time.universe, [0, 0, 1, 2])
washing_time['Normal'] = fuzz.trapmf(washing_time.universe, [1, 2, 2, 3])
washing_time['Extended'] = fuzz.trapmf(washing_time.universe, [2, 3, 4, 4])

# Define fuzzy rules (Inference)
rule1 = ctrl.Rule(dirt_level['Lightly_Soiled'] & load_size['Small'], washing_time['Short'])
rule2 = ctrl.Rule(dirt_level['Lightly_Soiled'] & load_size['Medium'], washing_time['Normal'])
rule3 = ctrl.Rule(dirt_level['Lightly_Soiled'] & load_size['Large'], washing_time['Normal'])

rule4 = ctrl.Rule(dirt_level['Moderately_Soiled'] & load_size['Small'], washing_time['Short'])
rule5 = ctrl.Rule(dirt_level['Moderately_Soiled'] & load_size['Medium'], washing_time['Normal'])
rule6 = ctrl.Rule(dirt_level['Moderately_Soiled'] & load_size['Large'], washing_time['Extended'])

rule7 = ctrl.Rule(dirt_level['Heavily_Soiled'] & load_size['Small'], washing_time['Normal'])
rule8 = ctrl.Rule(dirt_level['Heavily_Soiled'] & load_size['Medium'], washing_time['Extended'])
rule9 = ctrl.Rule(dirt_level['Heavily_Soiled'] & load_size['Large'], washing_time['Extended'])

# Create a control system and define rules (Inference)
washing_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])

# Create a simulation with the control system (Inference)
washing_machine = ctrl.ControlSystemSimulation(washing_ctrl)

# Visualize the membership functions and control surface
dirt_level.view()
load_size.view()
washing_time.view()

# Input values for dirt level and load size
washing_machine.input['Dirt_Level'] = int(input("Enter the scale of the dirt from 0 - 10: "))
washing_machine.input['Load_Size'] = int(input("Enter load size percentage from 0 - 100: "))

# Compute the output (Defuzzification)
washing_machine.compute()

# View the output
print("\nEstimated Washing Time in hours:", washing_machine.output['Washing_Time'])

washing_time.view(sim = washing_machine)

input("\nPress to exit.....")