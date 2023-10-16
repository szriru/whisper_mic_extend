import logging
import re
from typing_extensions import Literal
from rich.logging import RichHandler


def get_logger(name: str, level: Literal["info", "warning", "debug"]) -> logging.Logger:
    rich_handler = RichHandler(level=logging.INFO, rich_tracebacks=True, markup=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging._nameToLevel[level.upper()])

    if not logger.handlers:
        logger.addHandler(rich_handler)

    logger.propagate = False

    return logger

# for loading a fine-tuned whisper model.abs
# https://github.com/openai/whisper/discussions/830#discussioncomment-4652413
def hf_to_whisper_states(text):
    text = re.sub('.layers.', '.blocks.', text)
    text = re.sub('.self_attn.', '.attn.', text)
    text = re.sub('.q_proj.', '.query.', text)
    text = re.sub('.k_proj.', '.key.', text)
    text = re.sub('.v_proj.', '.value.', text)
    text = re.sub('.out_proj.', '.out.', text)
    text = re.sub('.fc1.', '.mlp.0.', text)
    text = re.sub('.fc2.', '.mlp.2.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.encoder_attn.', '.cross_attn.', text)
    text = re.sub('.cross_attn.ln.', '.cross_attn_ln.', text)
    text = re.sub('.embed_positions.weight', '.positional_embedding', text)
    text = re.sub('.embed_tokens.', '.token_embedding.', text)
    text = re.sub('model.', '', text)
    text = re.sub('attn.layer_norm.', 'attn_ln.', text)
    text = re.sub('.final_layer_norm.', '.mlp_ln.', text)
    text = re.sub('encoder.layer_norm.', 'encoder.ln_post.', text)
    text = re.sub('decoder.layer_norm.', 'decoder.ln.', text)
    return text