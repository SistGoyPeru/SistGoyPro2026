import logging
import threading
from django.apps import AppConfig

logger = logging.getLogger(__name__)


class PredictorConfig(AppConfig):
    name = 'predictor'

    def ready(self):
        """Pre-calienta todos los modelos en un hilo de fondo para que el primer
        request no tenga que entrenarlos desde cero y no provoque timeout."""
        def _warmup():
            from predictor.engine import (
                get_prediction_service_spain,
                get_prediction_service_bundesliga,
                get_prediction_service_premier,
                get_prediction_service_seriea,
                get_prediction_service_ligue1,
                get_prediction_service_primeiraliga,
                get_prediction_service_proleague,
                get_prediction_service_eredivisie,
            )
            factories = [
                ("spain",        get_prediction_service_spain),
                ("bundesliga",   get_prediction_service_bundesliga),
                ("premier",      get_prediction_service_premier),
                ("seriea",       get_prediction_service_seriea),
                ("ligue1",       get_prediction_service_ligue1),
                ("primeiraliga", get_prediction_service_primeiraliga),
                ("proleague",    get_prediction_service_proleague),
                ("eredivisie",   get_prediction_service_eredivisie),
            ]
            for name, factory in factories:
                try:
                    factory()
                    logger.info("[warmup] %s listo", name)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[warmup] %s falló: %s", name, exc)

        thread = threading.Thread(target=_warmup, daemon=True, name="model-warmup")
        thread.start()
