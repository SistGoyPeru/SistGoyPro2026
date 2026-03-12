from django.shortcuts import render

from .engine import (
	get_prediction_service_spain,
	get_prediction_service_bundesliga,
	get_prediction_service_premier,
	get_prediction_service_seriea,
	get_prediction_service_ligue1,
)


def dashboard(request):
	liga = request.GET.get("liga") or request.POST.get("liga", "spain")
	if liga not in ("spain", "bundesliga", "premier", "seriea", "ligue1"):
		liga = "spain"

	if liga == "bundesliga":
		service = get_prediction_service_bundesliga()
	elif liga == "premier":
		service = get_prediction_service_premier()
	elif liga == "seriea":
		service = get_prediction_service_seriea()
	elif liga == "ligue1":
		service = get_prediction_service_ligue1()
	else:
		service = get_prediction_service_spain()
	fixtures = service.get_pending_fixtures()
	selected_match_key = request.POST.get("match_key") if request.method == "POST" else ""

	prediction = None
	error_message = ""
	if fixtures:
		if not selected_match_key:
			selected_match_key = fixtures[0]["match_key"]
		try:
			prediction = service.predict_match(selected_match_key)
		except ValueError as exc:
			error_message = str(exc)
	else:
		error_message = "No hay encuentros pendientes en el calendario actual."

	context = {
		"liga": liga,
		"league_name": service.league_name,
		"league_logo_url": service.league_logo_url,
		"datasets": service.dataset_labels,
		"fixtures": fixtures,
		"prediction": prediction,
		"selected_match_key": selected_match_key,
		"model_name": service.best_model_name,
		"validation_accuracy": round(service.validation_accuracy * 100, 2),
		"model_scores": service.model_scores,
		"error_message": error_message,
	}
	return render(request, "predictor/dashboard.html", context)
