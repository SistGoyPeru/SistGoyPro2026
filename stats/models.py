from django.db import models

# Create your models here.
class Match(models.Model):
    date = models.DateField()
    home_team = models.CharField(max_length=100)
    away_team = models.CharField(max_length=100)
    
    # Goals
    fthg = models.IntegerField(verbose_name="Full Time Home Goals")
    ftag = models.IntegerField(verbose_name="Full Time Away Goals")
    
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
