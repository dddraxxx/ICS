import transformers
from model.ICSA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset_ics import HybridDataset, ValDataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         IMAGE_TOKEN_INDEX, IGNORE_INDEX,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)

model_paths = ([
    "liuhaotian/llava-v1.5-7b",
    "liuhaotian/llava-v1.5-13b",
])

for model_path in model_paths:
    print("Loading model from {}".format(model_path))
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=None,
        padding_side="right",
        use_fast=False,
        legacy=True
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=None,
        output_hidden_states=True,
        output_attentions=True,
        use_cache=True,
    )
    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     model_path=model_path,
    #     model_base=None,
    #     model_name=get_model_name_from_path(model_path)
    # )
