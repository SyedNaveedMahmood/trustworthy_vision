"""Conservative keyed consensus for actual independent reviewer CSVs."""
import collections

VALID={'yes','no','uncertain','not_applicable',''}


def assess_reviews(key,ratings):
    lookup={r['review_id']:r for r in key};cells={};raters=set()
    if len(lookup)!=len(key):raise ValueError('Duplicate blinded packet IDs')
    for r in ratings:
        rid=r.get('reviewer_id','').strip();item=r.get('review_id','')
        if not rid:continue
        if item not in lookup:raise ValueError('Unknown review packet')
        cell=(item,rid)
        if cell in cells:raise ValueError('Duplicate reviewer/packet judgment')
        for field in ['removal_validity','sham_preservation','off_target_validity']:
            if r.get(field,'') not in VALID:raise ValueError('Unknown human validity value')
        confidence=r.get('confidence_1_to_5','')
        if confidence and confidence not in ['1','2','3','4','5']:raise ValueError('Invalid reviewer confidence')
        cells[cell]=r;raters.add(rid)
    if len(raters)<2:return {'status':'HUMAN_AUDIT_PENDING','independent_rater_ids_declared':sorted(raters),'submitted_judgment_rows':len(cells),'passed':False,'groups':{}}
    if len(raters)>2:raise ValueError('Declare fixed two-rater panel before adjudication; do not choose favorable rater pairs')
    raters=sorted(raters);groups=collections.defaultdict(list)
    for item,k in lookup.items():
        role='candidate' if k['region_index']<2 else 'independent_control';condition=k['condition']
        field='off_target_validity' if role=='independent_control' else 'removal_validity' if condition=='removal' else 'sham_preservation'
        values=[cells.get((item,rid),{}).get(field,'') for rid in raters]
        groups[(k['operator'],role,condition,field)].append(values)
    expected={(op,role,condition,'off_target_validity' if role=='independent_control' else 'removal_validity' if condition=='removal' else 'sham_preservation') for op in ['occlusion','blur','lama'] for role in ['candidate','independent_control'] for condition in ['removal','sham']}
    missing=sorted(expected-set(groups));results={};all_complete=not missing;all_pass=not missing
    for key,values in sorted(groups.items()):
        completed=[v for v in values if all(x!='' for x in v)];n=len(values);threshold=.90 if key[3]=='removal_validity' else .95
        passed=sum(all(x=='yes' for x in v) for v in values);complete=len(completed)==n and n>=30
        results['/'.join(key)]={'packets':n,'fully_rated_packets':len(completed),'both_yes':passed,'consensus_validity_rate':passed/n,
                              'disagreements':sum(a!=b for a,b in completed),'uncertain_or_na_packets':sum(any(x in ['uncertain','not_applicable'] for x in v) for v in completed),
                              'threshold':threshold,'complete':complete,'passed':complete and passed/n>=threshold}
        all_complete &=complete;all_pass &=complete and passed/n>=threshold
    return {'status':'PASS_HUMAN_CONSENSUS' if all_pass else 'FAILED_HUMAN_CONSENSUS' if all_complete else 'HUMAN_AUDIT_PENDING',
            'passed':all_pass,'missing_operator_condition_groups':['/'.join(x) for x in missing],'independent_rater_ids_declared':raters,'submitted_judgment_rows':len(cells),'groups':results,
            'limitation':'Distinct declared IDs are necessary; software cannot authenticate human identity or independence. Region-reference completeness needs separate review; this alone never locks E5.'}
