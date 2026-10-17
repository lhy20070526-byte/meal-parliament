"""The five specialists and the chair used by Meal Parliament."""

from .nutritionist import Nutritionist
from .wallet import Wallet
from .lazy import Lazy
from .fridge import Fridge
from .taste import Taste
from .speaker import Speaker

SPECIALISTS = (Nutritionist, Wallet, Lazy, Fridge, Taste)

__all__ = ["Nutritionist", "Wallet", "Lazy", "Fridge", "Taste", "Speaker", "SPECIALISTS"]
