from typing import Optional

from modal import gpu, method
from modal.cls import ClsMixin
import json

from .common import (
    MODEL_PATH,
    # generate_prompt,
    output_vol,
    stub,
    VOL_MOUNT_PATH,
    # user_model_path,
)

from .common_sql import (
    user_data_path,
    user_model_path,
    generate_prompt_sql,
)


@stub.cls(
    gpu=gpu.A100(memory=20),
    network_file_systems={VOL_MOUNT_PATH: output_vol},
)
class OpenLlamaModel(ClsMixin):
    def __init__(self, user: str, team_id: Optional[str] = None):
        import sys

        import torch
        from peft import PeftModel
        from transformers import LlamaForCausalLM, LlamaTokenizer

        self.user = user
        CHECKPOINT = user_model_path(self.user)

        load_8bit = False
        device = "cuda"

        self.tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)

        model = LlamaForCausalLM.from_pretrained(
            MODEL_PATH,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        model = PeftModel.from_pretrained(
            model,
            CHECKPOINT,
            torch_dtype=torch.float16,
        )

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        self.model = model
        self.device = device

    @method()
    def generate(
        self,
        input: str,
        context: str,
        max_new_tokens=128,
        **kwargs,
    ):
        import torch
        from transformers import GenerationConfig

        prompt = generate_prompt_sql(self.user, input, context)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        # print(tokens)
        generation_config = GenerationConfig(
            **kwargs,
        )
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s, skip_special_tokens=True)
        return output.split("### Response:")[1].strip()


@stub.local_entrypoint()
def main(user: str):
    # inputs = [
    #     "Tell me about alpacas.",
    #     "Tell me about the president of Mexico in 2019.",
    #     "What should we do next? Who should work on this?",
    #     "What are your political views?",
    #     "What did you work on yesterday?",
    #     "@here is anyone in the office?",
    #     "What did you think about the last season of Silicon Valley?",
    #     "Who are you?",
    # ]

    # load data
    fp = open("src/data/test_data.jsonl", "r")
    eval_data = [json.loads(line) for line in fp]

    model = OpenLlamaModel.remote(user, None)
    for row_dict in eval_data:
        print("Input: " + str(row_dict))
        print(
            model.generate(
                row_dict["input"],
                row_dict["context"],
                do_sample=True,
                temperature=0.3,
                top_p=0.85,
                top_k=40,
                num_beams=1,
                max_new_tokens=600,
                repetition_penalty=1.2,
            )
        )
