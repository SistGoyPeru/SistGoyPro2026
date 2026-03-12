# Generated manually for fixture link cache

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("predictor", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="FixtureLinkCache",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("league_key", models.CharField(max_length=24)),
                ("match_key", models.CharField(max_length=255)),
                ("competition", models.CharField(blank=True, default="", max_length=120)),
                ("round_name", models.CharField(blank=True, default="", max_length=120)),
                ("match_date", models.CharField(blank=True, default="", max_length=40)),
                ("match_time", models.CharField(blank=True, default="", max_length=20)),
                ("home_team", models.CharField(max_length=120)),
                ("away_team", models.CharField(max_length=120)),
                ("result", models.CharField(blank=True, default="", max_length=20)),
                ("status", models.CharField(blank=True, default="", max_length=60)),
                ("match_link", models.URLField(blank=True, default="")),
                ("fetched_at", models.DateTimeField(default=django.utils.timezone.now)),
            ],
            options={
                "verbose_name": "Caché de enlace de encuentro",
                "verbose_name_plural": "Caché de enlaces de encuentros",
                "unique_together": {("league_key", "match_key")},
            },
        ),
    ]
