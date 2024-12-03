from model.methods.base import Method
from model.models.tabllm import TabularLLaMA
from model.models.ftt import Transformer

class TabLLMMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy == 'indices')

    def construct_model(self, model_config = None):
        if model_config is None:
            model_config = self.args.config['model']
        self.base_model = Transformer(
            d_numerical=self.d_in,
            categories=self.categories,
            d_out=model_config['llm_model']['base_output_dim'],
            **model_config['base_model']
        )
        self.model = TabularLLaMA(
            base_model=self.base_model,
            d_out=self.d_out,
            **model_config['llm_model']
        )
        self.model = self.model.to(self.args.device)
        


