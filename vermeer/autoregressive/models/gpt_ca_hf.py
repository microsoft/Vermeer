from autoregressive.models.gpt import ModelArgs, Transformer
from huggingface_hub import PyTorchModelHubMixin


class TransformerHF(Transformer, PyTorchModelHubMixin, repo_url="https://github.com/microsoft/vermeer/", license="mit", tags=["vermeer"]):
    pass


#################################################################################
#                                GPT Configs                                    #
#################################################################################
def Vermeer_XL(**kwargs):
    return TransformerHF(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M

def Vermeer_L(**kwargs):
    return TransformerHF(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)) # 343M

def Vermeer_B(**kwargs):
    return TransformerHF(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) # 111M
        

Vermeer_models_HF = {
    'Vermeer-B': Vermeer_B, 'Vermeer-L': Vermeer_L, 'Vermeer-XL': Vermeer_XL,
}
