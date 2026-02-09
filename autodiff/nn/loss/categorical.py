from autodiff.node import Node

def categorical_cross_entropy(logits:Node, one_hot_labels:Node):
    """
    logits: (batch_size, num_classes) raw scores
    one_hot_labels: (batch_size, num_classes) one-hot encoded ground truth
    """
    # Step 1: Shift logits for numerical stability (only +, -, exp, sum used)
    shifted_logits = logits - logits.max(1).unsqueeze(1)   # subtract for stability
    
    # Step 2: Compute exp for each logit
    exp_vals = shifted_logits.exp()                        # primitive exp
    
    # Step 3: Compute softmax probabilities
    softmax_probs = exp_vals / exp_vals.sum(1).unsqueeze(1) # primitive division
    
    # Step 5: Take log of probabilities
    log_probs = softmax_probs.ln()                     # primitive log
    
    # Step 6: Select the log prob of the correct class using one-hot masking
    correct_logprobs = -1 * (one_hot_labels * log_probs).sum(1).unsqueeze(1)
    
    # Step 7: Take mean over batch
    loss = correct_logprobs.mean(-1)
    return loss
