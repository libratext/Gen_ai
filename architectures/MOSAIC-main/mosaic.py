from typing import List, Optional
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

torch.set_grad_enabled(False)

def apply_top_p_with_epsilon(logits: torch.Tensor, top_p: float, epsilon: float = 1e-10) -> torch.Tensor:
    """
    Applies a top-p (nucleus) filtering to logits but, instead of setting
    the logits of non-selected tokens to -inf (which would result in zero probability),
    sets them to log(epsilon), so that the support remains the same.
    
    Parameters:
      logits: Tensor of shape (batch, seq_len, vocab_size)
      top_p: The nucleus threshold (e.g. 0.7, 0.8, etc.)
      epsilon: The small value to assign to tokens not selected.
      
    Returns:
      new_logits: Tensor with the same shape as logits.
    """
    # Compute probabilities from logits
    probs = F.softmax(logits, dim=-1)
    # Sort probabilities (descending) along the vocabulary dimension.
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    # Compute the cumulative sum along the sorted probabilities.
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # Create a mask: True for tokens to keep.
    # We keep tokens until cumulative_probs <= top_p.
    keep_mask = cumulative_probs <= top_p

    # Ensure that at least one token is kept per example: if none are kept, keep the top one.
    # Here we check along the vocab dimension.
    no_token_kept = keep_mask.sum(dim=-1, keepdim=True) == 0
    if no_token_kept.any():
        # For positions where no token was kept, set the first token (highest probability) to True.
        # Note: torch.scatter_ returns a modified tensor.
        # We create a tensor of zeros (False) and then scatter True into the first column.
        fix_mask = torch.zeros_like(keep_mask, dtype=torch.bool)
        fix_mask.scatter_(-1, torch.zeros_like(keep_mask[..., :1], dtype=torch.long), True)
        keep_mask = torch.where(no_token_kept, fix_mask, keep_mask)

    # Now, create new logits: copy the original logits.
    new_logits = logits.clone()
    # For tokens that are not kept (i.e. where keep_mask is False), set their logit to log(epsilon)
    new_logits[~keep_mask] = torch.log(torch.tensor(epsilon, device=logits.device, dtype=logits.dtype))
    return new_logits

class Mosaic(object):
    def __init__(self,
                 model_name_or_paths: List[str],
                 use_bfloat16: bool = True,
                 max_token_observed: int = 512,
                 unigram: Optional[str] = None,
                 custom_config : Optional[List[bool]] = None,
                 stupid_mode: bool = False,
                 one_model_mode: bool = False
                 ) -> None:
        self.models = []
        for i, model_name_or_path in enumerate(model_name_or_paths):
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                         device_map="auto",
                                                         trust_remote_code=True,
                                                         torch_dtype=torch.bfloat16 if use_bfloat16
                                                         else torch.float32
                                                         )
            model.eval()  # Set the model to evaluation mode
            self.models.append(model)
            print(f"Loaded model: {model_name_or_path}")
            # Print the device map
            #print(f"Device map for {model_name_or_path}: {model.hf_device_map}")
        
        if stupid_mode:
            self.max_iters = 0
        else:
            self.max_iters = 1000

        self.one_model_mode = one_model_mode

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_paths[-1])
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_token_observed = max_token_observed

        self.nb_models = len(self.models)
        self.unigram_path = unigram

        if custom_config is None:
            custom_config = [False] * self.nb_models
        self.custom_config = custom_config

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False)
        return encodings
    
    def trim_logits(self, logits, max_length=32000):
        # Check the shape of the logits tensor
        if logits.shape[2] > max_length:
            # Slice the tensor to keep only the first max_length elements along the last dimension
            logits = logits[:, :, :max_length]
        return logits
    
    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> List[torch.Tensor]:
        # If one_model_mode is active, we simulate multiple models by applying top-p with different thresholds.
        if self.one_model_mode:
            # Compute base logits from the single model.
            model = self.models[0]
            device = next(model.parameters()).device
            model_encodings = encodings.to(device)
            base_logits = model(**model_encodings).logits
            # Optionally trim logits:
            # base_logits = self.trim_logits(base_logits)
            # Define the top-p thresholds (e.g., four different values)
            top_p_values = [0.7, 0.8, 0.9, 0.95]
            # Epsilon value for non-selected tokens (you can adjust this if needed)
            epsilon = 1e-10
            logits_list = []
            for top_p in top_p_values:
                warped_logits = apply_top_p_with_epsilon(base_logits, top_p, epsilon)
                logits_list.append(warped_logits)
        else:
            # Normal mode: use each model in self.models.
            logits_list = []
            for i, model in enumerate(self.models):
                device = next(model.parameters()).device
                model_encodings = encodings.to(device)
                logits = model(**model_encodings).logits
                # Optionally trim logits:
                # logits = self.trim_logits(logits)
                logits_list.append(logits)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
        
        if self.unigram_path:
            batch_size, seq_len, voc_size = logits_list[0].shape
            unigram_proba = torch.load(self.unigram_path)
            unigram_proba += 1e-10
            unigram_logits = torch.log(unigram_proba)
            # Optionally center logits if needed:
            logits = logits_list[0] - logits_list[0].mean(dim=-1, keepdim=True)
            expanded_unigram_logits = unigram_logits.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, voc_size)
            logits_list.append(expanded_unigram_logits)
        return logits_list
    
    def get_softmax_probabilities(self, input_text):
        encodings = self._tokenize(input_text)
        logits_list = self._get_logits(encodings)
        probabilities_list = softmax_probabilities_all_models(logits_list)
        return encodings, logits_list, probabilities_list
     
    def compute_arimoto_torch(self, input_text, max_iters=1000):
        encodings, logits_list, tensors_list = self.get_softmax_probabilities(input_text)
        nb_models = len(tensors_list)
        seq_len = len(encodings.input_ids[0])
        voc_size = tensors_list[0].shape[-1]

        device = tensors_list[0].device
        # Move all tensors in tensors_list to the device of the first tensor
        tensors_list = [tensor.to(device) for tensor in tensors_list]

        # Stack all model predictions along a new dimension to form a (seq_len, nb_models, voc_size) tensor
        probabilities_tensor = torch.stack([t[0] for t in tensors_list], dim=1).to(tensors_list[0].device)

        # Run the Blahut-Arimoto algorithm on the entire batch
        capacity, p = blahut_arimoto_torch(probabilities_tensor, max_iters=max_iters)
        
        # Prepare the weighted sum tensor, initially zeros
        weighted_sum_tensor = torch.zeros_like(tensors_list[0])

        # Here, we need an additional mechanism if 'p' shapes or logic require different handling
        # Assuming 'p' is now (seq_len, nb_models), apply weights to each model's output
        for i in range(nb_models):
            weighted_sum_tensor += p[:, i:i+1] * tensors_list[i]

        return encodings, weighted_sum_tensor, tensors_list, p, logits_list
    
    def compute_scores(self, input_text):
        encodings, weighted_sum_tensor, probabilities_list, arimoto_weights, logits_list = self.compute_arimoto_torch(input_text, max_iters=self.max_iters)
        log_ppl, ppl, nll = perplexity(encodings, weighted_sum_tensor)
        ppl_list = perplexity_all_models(encodings, logits_list)
        x_ppl_list = cross_entropy(weighted_sum_tensor, probabilities_list)
        return log_ppl, x_ppl_list, arimoto_weights, nll, ppl_list
    
    def compute_end_score(self, input_text):
        encodings, weighted_sum_tensor, probabilities_list, arimoto_weights, logits_list = self.compute_arimoto_torch(input_text)
        log_ppl, ppl, nll = perplexity(encodings, weighted_sum_tensor)
        ppl_list = perplexity_all_models(encodings, logits_list)
        x_ppl_list = cross_entropy(weighted_sum_tensor, probabilities_list)
        log_ppl_value = log_ppl.item()
        x_ppl_values = [x.item() for x in x_ppl_list]
        final_score = log_ppl_value - x_ppl_values[0] #Ensure your "reference model" is given as first argument
        return  final_score

def perplexity(encodings, weighted_sum_tensor):
    shifted_probabilities = weighted_sum_tensor[..., :-1, :].contiguous()
    shifted_labels = encodings.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encodings.attention_mask[..., 1:].contiguous()

    device = shifted_probabilities.device

    # Ensure all tensors are moved to the same device
    shifted_probabilities = shifted_probabilities.to(device)
    shifted_labels = shifted_labels.to(device)
    shifted_attention_mask = shifted_attention_mask.to(device)

    actual_next_token_probabilities = torch.gather(shifted_probabilities, 2, shifted_labels.unsqueeze(-1)).squeeze(-1)

    nll = -torch.log(actual_next_token_probabilities + 1e-12)
    nll_masked = nll * shifted_attention_mask

    # Calculate the average NLL per sequence, taking into account only the valid (non-padded) tokens
    average_nll = torch.sum(nll_masked, dim=1) / torch.sum(shifted_attention_mask, dim=1)

    # Calculate perplexity per sequence
    perplexity = torch.exp(average_nll)
    return average_nll, perplexity, nll_masked

def cross_entropy(weighted_sum_tensor, probabilities_list):
    device = weighted_sum_tensor.device
    x_ppl_list = []

    # Compute log of weighted_sum_tensor outside the loop since it doesn't depend on m2_probabilities
    log_M1 = torch.log(weighted_sum_tensor).to(device)

    for m2_probabilities in probabilities_list:
        m2_probabilities = m2_probabilities.to(device)
        # Ensure m2_probabilities is correctly shaped for batch matrix multiplication
        # log_M1 shape is already (batch_size, sequence_length, vocabulary_size)
        # We need m2_probabilities in shape (batch_size, vocabulary_size, sequence_length) for bmm
        m2_probabilities_transposed = m2_probabilities.transpose(1, 2)
        
        # Perform batch matrix multiplication
        # Resulting shape: (batch_size, sequence_length, sequence_length)
        # We sum over the vocabulary dimension, effectively computing the dot product for each sequence position
        dot_products = torch.bmm(log_M1, m2_probabilities_transposed)
        
        # Since we're interested in the diagonal (dot products of corresponding vectors), we extract it
        # The diagonal for each item in the batch gives us the dot products we're interested in
        # torch.diagonal doesn't support batched operations directly, so we need to workaround
        dot_products_diagonal = torch.einsum('bii->bi', dot_products)  # Using einsum to extract diagonals for batch
        
        # Compute the mean of the dot_products_diagonal across the sequence dimension
        # This gives us the average dot product per sequence, which is then negated
        x_ppl = -torch.mean(dot_products_diagonal, dim=1)
        
        x_ppl_list.append(x_ppl)
    x_ppl_tensor = torch.stack(x_ppl_list)
    return x_ppl_list #, x_ppl_tensor

def softmax_probabilities_all_models(logits_list: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Calculates the softmax probabilities for the entire sequence of tokens for each model.

    Parameters:
    - logits_list: List[torch.Tensor]
        A list containing the logits tensor for each model.

    Returns:
    - List[torch.Tensor]: A list of tensors, where each tensor is the softmax probabilities
      for one model across the entire sequence of tokens.
    """
    softmax_fn = torch.nn.Softmax(dim=-1)
    probabilities_list = []

    for logits in logits_list:
        # Calculate softmax probabilities across the vocabulary for each token position
        softmax_probabilities = softmax_fn(logits)
        probabilities_list.append(softmax_probabilities)

    return probabilities_list

def perplexity_logits(encoding, logits):
    # Ensure encoding tensors are moved to the same device as logits
    device = logits.device
    logits = torch.clamp(logits, min=-20, max=50)

    encoding_input_ids = encoding.input_ids.to(device)
    encoding_attention_mask = encoding.attention_mask.to(device)

    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = encoding_input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding_attention_mask[..., 1:].contiguous()

    # Calculate Cross-Entropy loss
    cross_entropy_loss = ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels)
    # Apply attention mask
    masked_ce_loss = cross_entropy_loss * shifted_attention_mask
    # Calculate perplexity
    ppl = masked_ce_loss.sum(1) / shifted_attention_mask.sum(1)
    # Move result to CPU and convert to numpy for further processing if needed
    ppl = ppl.to("cpu").float().numpy()

    return ppl

def perplexity_all_models(encoding, logits_list):
    ppl_list = []
    for logits in logits_list:
        ppl = perplexity_logits(encoding, logits)
        ppl_list.append(ppl)
    return ppl_list

def blahut_arimoto_torch(W, epsilon=1e-6, max_iters=1000):
    """
    Batch-process Blahut-Arimoto using PyTorch for multiple sequences.
    """
    seq_len, nb_models, voc_size = W.shape
    p = torch.full((seq_len, nb_models), 1.0 / nb_models, device=W.device, dtype=W.dtype)
    prod_exp = torch.ones((seq_len, nb_models), device=W.device, dtype=W.dtype)

    for _ in range(max_iters):
        # Calculate the marginal probabilities
        sum_p_w = torch.bmm(p.unsqueeze(1), W).squeeze(1)  # Resultant shape: (seq_len, voc_size)

        # Calculate normalized probabilities
        W_normalized = W / sum_p_w.unsqueeze(1)  # Broadcasting to shape (seq_len, nb_models, voc_size)
        
        # Avoid numerical issues with logarithms
        W_normalized[W_normalized == 0] = torch.finfo(W.dtype).eps
        log_term = torch.log(W_normalized)
        log_term[torch.isnan(log_term) | torch.isinf(log_term)] = 0

        # Compute product exponentials and update probabilities
        prod_exp = torch.exp(torch.sum(W * log_term, axis=2))  # Sum across voc_size
        p_new = (p * prod_exp) / torch.sum(p * prod_exp, dim=1, keepdim=True)

        # Check convergence
        if torch.max(torch.abs(p - p_new)) < epsilon:
            break
        p = p_new

    # Compute channel capacity
    capacity = torch.log(torch.sum(p * prod_exp, dim=1)) / torch.log(torch.tensor(2.0, device=W.device))
    return capacity, p