from fairseq.models import register_model_architecture
from fairseq.models.transformer import base_architecture

@register_model_architecture("transformer", "transformer_1x_v0")
# pre-defining certain parameters for regularly used models
def transformer_base_v0(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    base_architecture(args)

@register_model_architecture("transformer", "transformer_1x_v1")
def transformer_base_v1(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.use_stable_embedding = getattr(args, "use_stable_embedding", True)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    base_architecture(args)


@register_model_architecture("transformer", "transformer_2x")
def transformer_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_architecture(args)


@register_model_architecture("transformer", "transformer_2x_v0")
def transformer_big_v0(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    base_architecture(args)


@register_model_architecture("transformer", "transformer_2x_v1")
def transformer_big_v1(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.use_stable_embedding = getattr(args, "use_stable_embedding", True)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    base_architecture(args)


@register_model_architecture("transformer", "transformer_4x")
def transformer_huge(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1536)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1536)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_architecture(args)


@register_model_architecture("transformer", "transformer_4x_v0")
# pre-defining certain parameters for regularly used models
def transformer_huge_v0(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1536)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1536)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    base_architecture(args)


@register_model_architecture("transformer", "transformer_4x_v1")
# pre-defining certain parameters for regularly used models
def transformer_huge_v1(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1536)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1536)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.use_stable_embedding = getattr(args, "use_stable_embedding", True)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    base_architecture(args)


@register_model_architecture("transformer", "transformer_4x_rs")
# pre-defining certain parameters for regularly used models
def transformer_huge_rs(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1536)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_recurrent_stacking = getattr(args, "encoder_recurrent_stacking", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1536)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_recurrent_stacking = getattr(args, "decoder_recurrent_stacking", 6)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    base_architecture(args)


@register_model_architecture("transformer", "transformer_4x_rs_v1")
# pre-defining certain parameters for regularly used models
def transformer_huge_rs_v1(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1536)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_recurrent_stacking = getattr(args, "encoder_recurrent_stacking", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1536)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_recurrent_stacking = getattr(args, "decoder_recurrent_stacking", 6)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.use_stable_embedding = getattr(args, "use_stable_embedding", True)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    base_architecture(args)
