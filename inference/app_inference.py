import gradio as gr
from generation import predict

from glob import glob
file_name = sorted(glob("lightning_logs/*/checkpoints/*.ckpt"))

# def predict(ckpt, inputs, max_length, top_k, top_p, temperature):
    # return str([ckpt, inputs, max_length, top_k, top_p, temperature])

def MHBAMixerV2_generate_demo():
    with gr.Blocks(analytics_enabled=False) as mhbamixerv2_interface:
        gr.Markdown("\
                    <div align='center'> <h2> MHBAMixerV2 For Generation </span> </h2>")
        with gr.Row():
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="source_inputs"):
                    with gr.TabItem('input_text'):
                            input_text = gr.TextArea(label="input (required)")
        with gr.Row():
            with gr.Column(variant="panel"):
                with gr.Tabs(elem_id="settings"):
                    with gr.TabItem("Checkpoints"):
                        ckpt = gr.Dropdown(choices=file_name, multiselect=False, container=False)

        with gr.Row():
            with gr.Column(variant="panel"):
                with gr.Tabs(elem_id="settings"):
                    with gr.TabItem("general settings"):
                        with gr.Column(variant='panel'):
                            temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="temperature",  value=0.8)
                            top_k = gr.Slider(minimum=1, maximum=10, step=1, label="top_k", value=5)
                            top_p = gr.Slider(minimum=0.1, maximum=1.0, step=0.01, label="top_p", value=0.8)
                            max_length = gr.Slider(minimum=1, maximum=2048, step=1, label="max_lenght", value=256)
                            submit = gr.Button("Generate", elem_id="generate", variant='primary')

            with gr.Column(variant="panel"):
                with gr.Tabs(elem_id="outputs"):
                    with gr.TabItem("output_text"):
                        outputs = gr.TextArea(label="outputs")
        
        # ckpt, inputs, max_length, top_k, top_p, temperature
        submit.click(
            fn=predict,
            inputs=[
                ckpt,
                input_text,
                max_length,
                top_k,
                top_p,
                temperature
            ],
            outputs=[
                outputs
            ]
        )
    return mhbamixerv2_interface

if __name__ == "__main__":
    demo = MHBAMixerV2_generate_demo()
    demo.launch(server_name="127.0.0.1", server_port=12345)