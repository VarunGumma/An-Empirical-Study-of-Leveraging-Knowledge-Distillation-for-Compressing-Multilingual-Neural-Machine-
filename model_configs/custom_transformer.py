from fairseq.models import register_model_architecture
from fairseq.models.transformer import base_architecture


@register_model_architecture("transformer", "transformer_base")
def transformer_base(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_architecture(args)


@register_model_architecture("transformer", "transformer_base12L")
def transformer_base12L(args):
    args.encoder_layers = getattr("encoder_layers", 12)
    args.decoder_layers = getattr("decoder_layers", 12)
    transformer_base(args)


@register_model_architecture("transformer", "transformer_base18L")
def transformer_base18L(args):
    args.encoder_layers = getattr("encoder_layers", 18)
    args.decoder_layers = getattr("decoder_layers", 18)
    transformer_base(args)


@register_model_architecture("transformer", "transformer_base24L")
def transformer_base24L(args):
    args.encoder_layers = getattr("encoder_layers", 24)
    args.decoder_layers = getattr("decoder_layers", 24)
    transformer_base(args)


@register_model_architecture("transformer", "transformer_big")
def transformer_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    transformer_base(args)


@register_model_architecture("transformer", "transformer_it")
def transformer_it(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1536)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1536)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_architecture(args)


@register_model_architecture("transformer", "transformer_huge")
def transformer_huge(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    transformer_it(args)


@register_model_architecture("transformer", "transformer_huge_RS")
def transformer_huge(args):
    args.encoder_recurrent_stacking = getattr(args, "encoder_recurrent_stacking", 6)
    args.decoder_recurrent_stacking = getattr(args, "decoder_recurrent_stacking", 6)
    transformer_huge(args)
