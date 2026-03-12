from django.db import models
from django.utils import timezone


class InjuryCache(models.Model):
    """Caché persistente de bajas/lesiones para un partido."""

    home_team = models.CharField(max_length=120)
    away_team = models.CharField(max_length=120)
    source = models.CharField(max_length=60, default="")
    data = models.JSONField()
    fetched_at = models.DateTimeField(default=timezone.now)

    class Meta:
        unique_together = ("home_team", "away_team")
        verbose_name = "Caché de bajas"
        verbose_name_plural = "Caché de bajas"

    def __str__(self) -> str:
        return f"{self.home_team} vs {self.away_team} [{self.source}]"
