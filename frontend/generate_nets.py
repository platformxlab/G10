from statistics import mode
from torchvision.models import resnet18, resnet34, resnet50, resnet152, inception_v3, alexnet, wide_resnet101_2, resnext101_32x8d, wide_resnet50_2, densenet121
from senet import senet16, senet154
from inceptionresnetv2 import inceptionresnetv2
import binascii
# from dlrm import DLRM_Net
# from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2Config, GPT2Model, AutoConfig, AutoModelForCausalLM

# model1 = senet154()
# model2 = inceptionresnetv2()
# model3 = resnext101_32x8d()
# model4 = DLRM_Net()
model5 = AutoModel.from_pretrained("bert-base-uncased")


# Iterator for Training
def batch_iterator(batch_size=10):
    for _ in tqdm(range(0, args.n_examples, batch_size)):
        yield [next(iter_dataset)["content"] for _ in range(batch_size)]

# Base tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
base_vocab = list()


# Configuration
config_kwargs = {"vocab_size": len(tokenizer),
                 "scale_attn_by_layer_idx": True,
                 "reorder_and_upcast_attn": True}

# Load model with config and push to hub
config = AutoConfig.from_pretrained('gpt2-large', **config_kwargs)
model6 = AutoModelForCausalLM.from_config(config)





print(model5)