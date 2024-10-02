import torch
import transformers
from lib.model_extra import MultiHead, ModelOutput


class TransformerModel(torch.nn.Module):
    INPUT_SHAPE = (128,)
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.model = transformers.AutoModel.from_pretrained(args.model_name)
        self.output = MultiHead(args)

    def forward(self, batch):
        output = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'])

        representation = output.last_hidden_state.mean(axis = 1)

        model_output = ModelOutput(representation = representation)
        model_output = self.output(model_output)
        return model_output