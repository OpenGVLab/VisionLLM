CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

DEFAULT_TOKENS = {
    'pad': "[PAD]",
    'bos': "<s>",
    'eos': "</s>",
    'unk': "<unk>",
    # placeholder
    'img': "<image>",
    'imp': "<im_patch>",
    'reg': "<region>",
    # special tokens (start, end)
    'boi': "<img>",
    'eoi': "</img>",
    'sor': "<reg>",
    'eor': "</reg>",
    'sod': "<det>",
    'eod': "</det>",
    'sog': "<grd>",
    'eog': "</grd>",
    # tools
    'det': "[DET]",
    'grd': "[GRD]",
    'seg': "[SEG]",
    'pose': "[POSE]",
    'gen': "[GEN]",
    'edit': "[EDIT]",
    # embeddings
    'emb': "[EMB]",
    'emb2': "[EMB2]",
    'emb3': "[EMB3]",
    'emb4': "[EMB4]",
    'emb5': "[EMB5]",
    'emb6': "[EMB6]",
    'emb7': "[EMB7]",
    'emb8': "[EMB8]",
}
