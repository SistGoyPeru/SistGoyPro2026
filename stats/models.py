from django.db import models

# Create your models here.
class Match(models.Model):
    date = models.DateField()
    home_team = models.CharField(max_length=100)
    away_team = models.CharField(max_length=100)
    
    # Goals
    fthg = models.IntegerField(verbose_name="Full Time Home Goals")
    ftag = models.IntegerField(verbose_name="Full Time Away Goals")
    
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

    class Meta:
        verbose_name_plural = "Matches"
        ordering = ['-date']

    def __str__(self):
        return f"{self.date} - {self.home_team} {self.fthg}-{self.ftag} {self.away_team}"
