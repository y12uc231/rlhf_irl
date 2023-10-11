# Hypothesis : Can we retrieve reward function with existing IRL methods?
# Step 1: Fine-tune Pythia-70Mn reward model with toxic dataset (DT) - call it TRM (Target RM)
# Step 2: PPO Pythia-70Mn reward model with TRM
# Step 3: Take the pretrained reward model (Pythia-70Mn) with random weights to spit reward value – call it SRM (Source RM)
    # Apply IRL with SRM as the start RM. – Returns Approximated RM. 
# Step 4:  Evaluate Approximated RM. 
