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


class FixtureLinkCache(models.Model):
    """Caché persistente de encuentros y enlaces por liga."""

    league_key = models.CharField(max_length=24)
    match_key = models.CharField(max_length=255)
    competition = models.CharField(max_length=120, blank=True, default="")
    round_name = models.CharField(max_length=120, blank=True, default="")
    match_date = models.CharField(max_length=40, blank=True, default="")
    match_time = models.CharField(max_length=20, blank=True, default="")
    home_team = models.CharField(max_length=120)
    away_team = models.CharField(max_length=120)
    result = models.CharField(max_length=20, blank=True, default="")
    status = models.CharField(max_length=60, blank=True, default="")
    match_link = models.URLField(blank=True, default="")
    fetched_at = models.DateTimeField(default=timezone.now)

    class Meta:
        unique_together = ("league_key", "match_key")
        verbose_name = "Caché de enlace de encuentro"
        verbose_name_plural = "Caché de enlaces de encuentros"

    def __str__(self) -> str:
        return f"{self.league_key}: {self.home_team} vs {self.away_team}"
