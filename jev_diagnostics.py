"""Routing telemetry derived from validated enums/numbers, never raw user/source text."""
from __future__ import annotations

import json
import logging

from jev_interpreter import _choice, _noul

log = logging.getLogger('ars.jev.diagnostics')


def emit(trace, stage, **metrics):
    # Callers supply program-defined stages/reasons and validated numeric proof.
    # Never pass provider request/response bodies, labels, goals or exceptions.
    try:
        log.info('route_diagnostic %s', json.dumps(
            {'trace':trace,'stage':stage,**metrics}, allow_nan=False, sort_keys=True))
    except Exception:
        # Observability must not veto a valid decision or break cache handling.
        # Do not fall back to printing handler/provider exception contents.
        return


def emit_selection(trace, answers, questions, mapping, evidence, route):
    try:
        metrics=selection_metrics(answers,questions,mapping,evidence,route)
    except Exception as exc:
        # An unused malformed metric does not replace the actual routing guards.
        emit(trace,'selection',error=type(exc).__name__)
    else:
        emit(trace,'selection',**metrics)


def selection_metrics(answers, questions, mapping, evidence, route):
    chosen, probabilities = _choice(answers['matching_key'], questions['matching_key']['criteria'])
    status, status_probabilities = _choice(answers['route_status'], questions['route_status']['criteria'])
    pair, pair_probabilities = _choice(answers['clarification_pair'], questions['clarification_pair']['criteria'])
    selected=mapping.get(chosen)
    source={key:evidence.confidence_by_key.get(key,0.) for key in mapping.values()}
    outcomes={key:_noul(answers['outcome_'+{'*':'star','#':'hash'}.get(key,key)])
              for key in mapping.values()}
    service=_noul(answers['has_matching_service'])
    strength=min(answers['matching_key']['confidence'],probabilities[chosen])
    route_strength=min(answers['route_status']['confidence'],status_probabilities[status])
    failed=[]
    if status=='unknown':failed.append('model_unknown')
    if status=='proceed':
        if selected is None:failed.append('no_candidate_chosen')
        if route_strength<.7:failed.append('route_below_minimum')
        if strength<.7:failed.append('choice_below_minimum')
        if service<.7:failed.append('service_below_minimum')
    return {'choice':selected,'choice_strength':strength,'choice_probabilities':probabilities,
        'route':status,'route_strength':route_strength,'route_probabilities':status_probabilities,
        'service':service,'service_minimum':.7,'choice_minimum':.7,'route_minimum':.7,
        'source':source,'direct_outcomes':outcomes,
        'user_detail_required':_noul(answers['user_detail_required']),
        'clarification_pair':pair,'clarification_strength':min(answers['clarification_pair']['confidence'],pair_probabilities[pair]),
        'result':route.status,'failed':failed}
