import gradio as gr
from mosaic import Mosaic  # adjust import as needed

# Maximum number of model textboxes
MAX_MODELS = 10

def update_textboxes(n_visible):
    """
    Given the current visible count, increments it by 1 (up to MAX_MODELS)
    and returns updated visibility settings for all model textboxes.
    """
    if n_visible < MAX_MODELS:
        n_visible += 1
    # Create a list of update objects for each textbox: visible if its index is less than n_visible.
    updates = []
    for i in range(MAX_MODELS):
        if i < n_visible:
            updates.append(gr.update(visible=True))
        else:
            updates.append(gr.update(visible=False))
    return n_visible, *updates

def run_scoring(input_text, model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, threshold_choice, custom_threshold):
    """
    Collect all non-empty model paths, instantiate Mosaic, compute the score,
    and return a message based on the threshold.
    """
    model_paths = []
    for m in [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10]:
        if m.strip() != "":
            model_paths.append(m.strip())
    if len(model_paths) < 2:
        return "Please enter at least two model paths.", None, None
    # Choose threshold value
    if threshold_choice == "default":
        threshold = 0.0
    elif threshold_choice == "raid":
        threshold = 0.23
    elif threshold_choice == "custom":
        threshold = custom_threshold
    else:
        threshold = 0.0
    # Instantiate the Mosaic class with the selected model paths.
    mosaic_instance = Mosaic(model_name_or_paths=model_paths, one_model_mode=False)
    final_score = mosaic_instance.compute_end_score(input_text)
    if final_score < threshold:
        result_message = "This text was probably generated."
    else:
        result_message = "This text is likely human-generated."
    return result_message, final_score, threshold

with gr.Blocks() as demo:
    gr.Markdown("# MOSAIC Scoring App")
    with gr.Row():
        input_text = gr.Textbox(lines=10, placeholder="Enter text here...", label="Input Text")
    with gr.Column():
        gr.Markdown("### Model Paths (at least 2 required)")
        gr.Markdown("Order matters for model 1 only, the Reference model. Please use the one with the best perplexity on human texts. (The largest LLM if applicable.) GPT2 models are enough to detect easy prompts from chatgpt.")
        # State to keep track of the number of visible textboxes (starting with 2)
        n_models_state = gr.State(2)
        # Create 10 textboxes. We'll name them model1, model2, ..., model10.
        model1 = gr.Textbox(value="openai-community/gpt2-large", label="Model 1 Path ", visible=True)
        model2 = gr.Textbox(value="openai-community/gpt2-medium", label="Model 2 Path", visible=True)
        model3 = gr.Textbox(value="", label="Model 3 Path", visible=False)
        model4 = gr.Textbox(value="", label="Model 4 Path", visible=False)
        model5 = gr.Textbox(value="", label="Model 5 Path", visible=False)
        model6 = gr.Textbox(value="", label="Model 6 Path", visible=False)
        model7 = gr.Textbox(value="", label="Model 7 Path", visible=False)
        model8 = gr.Textbox(value="", label="Model 8 Path", visible=False)
        model9 = gr.Textbox(value="", label="Model 9 Path", visible=False)
        model10 = gr.Textbox(value="", label="Model 10 Path", visible=False)
        # Add a plus button to reveal one more textbox.
        plus_button = gr.Button("+", elem_id="plus_button")
        # When plus_button is clicked, update n_models_state and all model textboxes.
        plus_button.click(
            fn=update_textboxes,
            inputs=n_models_state,
            outputs=[n_models_state, model1, model2, model3, model4, model5, model6, model7, model8, model9, model10]
        )
    with gr.Row():
        threshold_choice = gr.Radio(choices=["default", "raid", "custom"], value="default", label="Threshold Choice")
        custom_threshold = gr.Number(value=0.0, label="Custom Threshold (if 'custom' selected)")
    with gr.Row():
        output_message = gr.Textbox(label="Result Message")
        output_score = gr.Number(label="Final Score")
        output_threshold = gr.Number(label="Threshold Used")
    run_button = gr.Button("Run Scoring")
    run_button.click(
        fn=run_scoring,
        inputs=[input_text, model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, threshold_choice, custom_threshold],
        outputs=[output_message, output_score, output_threshold]
    )

demo.launch()
