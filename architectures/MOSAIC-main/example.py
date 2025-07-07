from mosaic import Mosaic
from datasets import load_dataset

threshold = 0 # You need to define this yourself, 0 is a good default value as our score was built this way but it can be optimised if you have a dataset similar to the one you wish to test
# threshold = 0.2328 is the best "TPR - FPR" (Youden’s index) possible on the raid samples with the Tower and Llama models

model_list = ["TowerBase-13B-v0.1", "TowerBase-7B-v0.1", "Llama-2-7b-chat-hf", "Llama-2-7b-hf"]

#model_list = ["openai-community/gpt2-medium", "openai-community/gpt2"] #for a lightweight test

text = "Please paste the text you wish to test here"

#One example from RAID if you wish, generated with GPT3 and greedy sampling 
#text = "In recent years, artificial intelligence (AI) has been increasingly applied to medical imaging, with the potential to improve patient care. However, the use of AI in medicine raises important ethical and societal concerns, including the need to ensure that AI systems are trustworthy. To address these concerns, we convened a multidisciplinary expert panel to develop guiding principles and consensus recommendations for the development and use of trustworthy AI in future medical imaging. The panel identified four key principles for trustworthy AI in medical imaging: (1) beneficence, (2) non-maleficence, (3) respect for autonomy, and (4) justice. The panel also identified a number of specific recommendations in each of these areas. We believe that these principles and recommendations can serve as a foundation for further discussion and action on this important topic."

mosaic = Mosaic(model_list)

# Compute scores using the mosaic object
score = mosaic.compute_end_score(text)

if score < threshold: # High score means human, as the cross-entropy does not exceed the surprisal
    print("This text sample was probably generated")
else:
    print("This text sample wasn’t generated")