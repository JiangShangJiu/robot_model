"""在预先指定的多个模型/噪声场景上筛选同一条激励轨迹。"""
from dataclasses import asdict
from .screening import (
    ExcitationCriteria, ExcitationSelection, NoFeasibleExcitation,
    screen_excitation, trajectory_record,
)


def screen_robust_excitation(regressor_func, candidates, scenarios, preprocess, *,
                             criteria=None, progress=None, retention_multiplier=1):
    """scenarios 为 [(名称, runner)]；preprocess(data, traj) 仅处理测量。

    所有场景均通过才合格。条件数取各场景最大值，最小奇异值取最小值；
    不按均值掩盖失败场景。保留场景 0 的数据供辨识，不混合不同对象参数。
    """
    criteria = criteria or ExcitationCriteria()
    scenarios = list(scenarios)
    if len(scenarios) < 2 or len({name for name, _ in scenarios}) != len(scenarios):
        raise ValueError('稳健性筛选需要至少两个不同名称的场景')
    records, winner, best = [], None, float('inf')
    for index, traj in enumerate(candidates):
        rec = {'candidate_id': index, 'trajectory': trajectory_record(traj),
               'accepted': False, 'reasons': [], 'scenarios': [], 'measured': None, 'preview': None}
        primary = None
        for name, runner in scenarios:
            try:
                selection = screen_excitation(
                    regressor_func, [traj], runner, lambda data: preprocess(data, traj),
                    criteria=criteria, retention_multiplier=retention_multiplier)
                detail = selection.report['candidates'][0]
                if primary is None:
                    primary = selection
            except NoFeasibleExcitation as exc:
                detail = exc.report['candidates'][0]
            rec['scenarios'].append({'name': name, **detail})
            if rec['preview'] is None:
                rec['preview'] = detail['preview']
            if not detail['accepted']:
                rec['reasons'].extend(f'{name}: {reason}' for reason in detail['reasons'])
        rec['accepted'] = not rec['reasons']
        rec['passed_scenarios'] = sum(s['accepted'] for s in rec['scenarios'])
        if rec['accepted']:
            qualities = [s['measured'] for s in rec['scenarios']]
            rec['measured'] = {
                'condition_number': max(q['condition_number'] for q in qualities),
                'min_singular_value': min(q['min_singular_value'] for q in qualities),
                'samples': min(q['samples'] for q in qualities),
                'rank': min(q['rank'] for q in qualities),
                'n_parms': qualities[0]['n_parms'],
            }
            score = (rec['measured']['condition_number'] if criteria.objective == 'condition_number'
                     else -rec['measured']['min_singular_value'])
            if score < best:
                best, winner = score, (index, primary)
        records.append(rec)
        if progress:
            progress(rec)
    report = {
        'criteria': asdict(criteria), 'scenarios': [name for name, _ in scenarios],
        'aggregation': 'all scenarios must pass; rank by worst case',
        'retention_multiplier': retention_multiplier,
        'selected_candidate_id': None if winner is None else winner[0],
        'candidate_count': len(records), 'accepted_count': sum(r['accepted'] for r in records),
        'candidates': records,
    }
    if winner is None:
        raise NoFeasibleExcitation(report)
    selection = winner[1]
    return ExcitationSelection(selection.trajectory, selection.run, selection.data, report)
