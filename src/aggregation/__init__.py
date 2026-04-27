from .spa import SPAAggregator
from .flexlora import FlexLoRAAggregator
from .flora import FLoRAAggregator
from .fedavg_homo import HomoAggregator, HeteroPadAggregator

AGGREGATOR_REGISTRY = {
    "homo_r4":    HomoAggregator,
    "homo_r8":    HomoAggregator,
    "hetero_pad": HeteroPadAggregator,
    "flexlora":   FlexLoRAAggregator,
    "hetero_spa": SPAAggregator,
}
