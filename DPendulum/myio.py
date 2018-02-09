import numpy as np
import re

def outABHistory(filename, history, mode = 'w'):
    f = open(filename, "w")
    arr = history.transpose
    for col in range(0,history.shape[1],1):
        for timestep in history[:,col]:
            s = str("%.4f"%timestep)+('\t')
            f.write(s)
        f.write("\n")

    f.close()

    return True

def readNNet(filename):

    nneurons = 0

    w_string_number = 0

    Mode = "General";

    file = open(filename, "r")
    for line in file:
        result = re.match(r'Initial values:', line)
        if(Mode == "General" and result != None):
            Mode = "Values"
            continue

        result = re.match(r'Weights:', line)
        if (Mode == "General" and result != None):
            Mode = "Weights"
            continue

        if (Mode == "Values"):
            line = line.replace('*', '')
            str_values = line.split()
            nneurons = len(str_values)
            values = np.zeros(nneurons)
            weights = np.zeros((nneurons, nneurons))
            for i in range (0, nneurons, 1):
                values[i] = float(str_values[i])
            Mode = "General"
            continue

        if (Mode == "Weights"):
            str_values = line.split()
            for i in range(0, nneurons, 1):
                weights[w_string_number, i] = float(str_values[i])
            w_string_number+=1


    return values, weights





