from django.db import models

# Create your models here.
class Match(models.Model):
    date = models.DateField()
    home_team = models.CharField(max_length=100)
    away_team = models.CharField(max_length=100)
    
    # Goals
    fthg = models.IntegerField(verbose_name="Full Time Home Goals")
    ftag = models.IntegerField(verbose_name="Full Time Away Goals")
    hthg = models.IntegerField(verbose_name="Half Time Home Goals", default=0)
    htag = models.IntegerField(verbose_name="Half Time Away Goals", default=0)
    
    # Advanced Stats
    hs = models.IntegerField(verbose_name="Home Shots", default=0)
    as_shots = models.IntegerField(verbose_name="Away Shots", default=0)
    hst = models.IntegerField(verbose_name="Home Shots on Target", default=0)
    ast = models.IntegerField(verbose_name="Away Shots on Target", default=0)
    hc = models.IntegerField(verbose_name="Home Corners", default=0)
    ac = models.IntegerField(verbose_name="Away Corners", default=0)
    hf = models.IntegerField(verbose_name="Home Fouls", default=0)
    af = models.IntegerField(verbose_name="Away Fouls", default=0)
    hy = models.IntegerField(verbose_name="Home Yellow Cards", default=0)
    ay = models.IntegerField(verbose_name="Away Yellow Cards", default=0)
    hr = models.IntegerField(verbose_name="Home Red Cards", default=0)
    ar = models.IntegerField(verbose_name="Away Red Cards", default=0)
    
    # Results (H=Home Win, D=Draw, A=Away Win)
    RESULT_CHOICES = [
        ('H', 'Home Win'),
        ('D', 'Draw'),
        ('A', 'Away Win'),
    ]
    ftr = models.CharField(max_length=1, choices=RESULT_CHOICES, verbose_name="Full Time Result")
    htr = models.CharField(max_length=1, choices=RESULT_CHOICES, verbose_name="Half Time Result", default='D')

    # Betting Odds (Market averages indicate expectations)
    avg_h = models.FloatField(verbose_name="Average Home Win Odds", default=0.0)
    avg_d = models.FloatField(verbose_name="Average Draw Odds", default=0.0)
    avg_a = models.FloatField(verbose_name="Average Away Win Odds", default=0.0)
    
    # Over / Under 2.5 Goals (Bet365 metrics)
    b365_o25 = models.FloatField(verbose_name="Bet365 Over 2.5", default=0.0)
    b365_u25 = models.FloatField(verbose_name="Bet365 Under 2.5", default=0.0)

    class Meta:
        verbose_name_plural = "Matches"
        ordering = ['-date']

    def __str__(self):
        return f"{self.date} - {self.home_team} {self.fthg}-{self.ftag} {self.away_team}"
