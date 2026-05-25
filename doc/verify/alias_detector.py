"""
Alias detection module for discovering PyTorch -> Paddle API name mappings.
"""


class AliasDetector:
    """Auto-discover PyTorch to Paddle API alias mappings."""

    # Predefined alias rules: (rule_name, transform_func, confidence, description)
    ALIAS_RULES = [
        {
            "name": "strip_underscore_prefix",
            "transform": lambda x: x.lstrip("_") if x.startswith("_") else None,
            "confidence": "high",
            "description": "PyTorch internal variant (underscore prefix) -> standard",
        },
        {
            "name": "paddle_elementwise_alias",
            "transform": lambda x: {
                "add": "elementwise_add",
                "subtract": "elementwise_sub",
                "multiply": "elementwise_mul",
                "divide": "elementwise_div",
                "floor_divide": "elementwise_floordiv",
                "remainder": "elementwise_mod",
                "maximum": "elementwise_max",
                "minimum": "elementwise_min",
                "fmax": "elementwise_fmax",
                "fmin": "elementwise_fmin",
                "pow": "elementwise_pow",
                "heaviside": "elementwise_heaviside",
            }.get(x),
            "confidence": "high",
            "description": "Paddle elementwise_xxx -> PyTorch arithmetic",
        },
        {
            "name": "paddle_reduce_alias",
            "transform": lambda x: {
                "all": "reduce_all",
                "any": "reduce_any",
                "amax": "reduce_amax",
                "amin": "reduce_amin",
                "max": "reduce_max",
                "min": "reduce_min",
                "mean": "reduce_mean",
                "sum": "reduce_sum",
                "prod": "reduce_prod",
            }.get(x),
            "confidence": "high",
            "description": "Paddle reduce_xxx -> PyTorch reduction",
        },
        {
            "name": "common_naming_differences",
            "transform": lambda x: {
                "arange": "range",
                "range": "arange",
                "embedding": "lookup_table_v2",
                "flatten": "flatten_contiguous_range",
                "full": "fill_constant",
                "full_like": "fill_any_like",
                "fill": "fill_any",
                "grid_sampler": "grid_sample",
                "log_sigmoid": "logsigmoid",
                "numel": "size",
                "one_hot": "one_hot_v2",
                "topk": "top_k_v2",
                "batch_norm": "batch_norm",
                "layer_norm": "layer_norm",
                "instance_norm": "instance_norm",
                "group_norm": "group_norm",
                "dropout": "dropout",
                "relu": "relu",
                "gelu": "gelu",
                "swish": "swish",
                "hardswish": "hardswish",
                "hardtanh": "hardtanh",
                "leaky_relu": "leaky_relu",
                "elu": "elu",
                "selu": "selu",
                "celu": "celu",
                "softmax": "softmax",
                "log_softmax": "log_softmax",
                "cross_entropy": "cross_entropy_with_softmax",
                "mse_loss": "mse_loss",
                "l1_loss": "l1_loss",
                "nll_loss": "nll_loss",
                "smooth_l1_loss": "smooth_l1_loss",
                "huber_loss": "huber_loss",
                "binary_cross_entropy": "binary_cross_entropy_with_logits",
                "kl_div": "kl_div",
                "margin_ranking_loss": "margin_ranking_loss",
                "multi_margin_loss": "multi_margin_loss",
                "multilabel_margin_loss": "multilabel_margin_loss",
                "soft_margin_loss": "soft_margin_loss",
                "triplet_margin_loss": "triplet_margin_loss",
                "ctc_loss": "ctc_loss",
                "hinge_embedding_loss": "hinge_embedding_loss",
                "cosine_embedding_loss": "cosine_embedding_loss",
            }.get(x),
            "confidence": "medium",
            "description": "Common naming differences between PyTorch and Paddle",
        },
        {
            "name": "conv_transpose_alias",
            "transform": lambda x: {
                "conv_transpose1d": "conv1d_transpose",
                "conv_transpose2d": "conv2d_transpose",
                "conv_transpose3d": "conv3d_transpose",
            }.get(x),
            "confidence": "high",
            "description": "conv_transposeNd -> convNd_transpose",
        },
        {
            "name": "max_pool_with_indices_alias",
            "transform": lambda x: {
                "max_pool1d_with_indices": "max_pool1d_with_index",
                "max_pool2d_with_indices": "max_pool2d_with_index",
                "max_pool3d_with_indices": "max_pool3d_with_index",
            }.get(x),
            "confidence": "high",
            "description": "max_poolNd_with_indices -> max_poolNd_with_index",
        },
        {
            "name": "paddle_v2_suffix",
            "transform": lambda x: f"{x}_v2",
            "confidence": "low",
            "description": "Paddle _v2 suffix version",
        },
    ]

    def __init__(self, paddle_api_h_funcs: set):
        """
        Args:
            paddle_api_h_funcs: Set of function names available in Paddle api.h
        """
        self.paddle_funcs = paddle_api_h_funcs

    @classmethod
    def from_paddle_tracer(cls, paddle_tracer):
        """Create detector from a PaddleTracer instance."""
        return cls(paddle_tracer.get_all_api_h_funcs())

    def discover(self, torch_op: str) -> list:
        """
        Discover all possible Paddle alias candidates for a PyTorch op.

        Returns list of dicts:
        {
            "torch_api": str,
            "paddle_api": str,
            "rule": str,
            "confidence": str,  # "high" / "medium" / "low"
            "description": str
        }
        """
        candidates = []

        for rule in self.ALIAS_RULES:
            transformed = rule["transform"](torch_op)
            if transformed and transformed in self.paddle_funcs:
                # Avoid duplicates
                if not any(c["paddle_api"] == transformed for c in candidates):
                    candidates.append(
                        {
                            "torch_api": torch_op,
                            "paddle_api": transformed,
                            "rule": rule["name"],
                            "confidence": rule["confidence"],
                            "description": rule["description"],
                        }
                    )

        return candidates

    def batch_discover(self, torch_ops: list) -> list:
        """Batch discover alias mappings."""
        all_candidates = []
        for op in torch_ops:
            candidates = self.discover(op)
            all_candidates.extend(candidates)
        return all_candidates
