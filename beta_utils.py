import math

def calculate_beta():
    h_minutes = 60 # hour-wise frequency
    #input is h_minutes: h_minutes=60mins means the traded volume will influence the following 60 mins (exponential decay)
    h_hours = h_minutes / 60

    # Calculate beta using half-life transformation
    beta = math.log(2) / h_hours
    return beta
