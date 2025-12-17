from tqdm import tqdm
from sklearn.model_selection import train_test_split

def split_by_taxid(proteins_by_taxid, val_perc, min_proteins_in_taxid):
    val_set = set()
    traintest_set = set()

    unfrequent_taxid_proteins = set()

    #print('Creating random validation splits by taxon id')
    for taxid, proteins in proteins_by_taxid.items():
        if len(proteins) < min_proteins_in_taxid:
            unfrequent_taxid_proteins.update(proteins)
        else:
            #val_n = int(round(len(proteins)*val_perc))
            ids_train, ids_validation = train_test_split(
                proteins, test_size = val_perc)
            val_set.update(ids_validation)
            traintest_set.update(ids_train)
    
    #print('Splitting', len(unfrequent_taxid_proteins), 'proteins in unfrequent taxa')
    if len(unfrequent_taxid_proteins) > 0:
        ids_train, ids_validation = train_test_split(list(unfrequent_taxid_proteins), 
            test_size = val_perc)
        val_set.update(ids_validation)
        traintest_set.update(ids_train)

    return traintest_set, val_set

def sep_validation(ids, taxids, proteins, val_perc, annotations):
    print('Validation percent:', val_perc)
    print('Grouping', len(proteins) , 'proteins by species')
    proteins_by_taxid = {taxid: [] for taxid in set(taxids)}
    proteins_set = set(proteins)
    for i in tqdm(range(len(ids)), total = len(ids)):
        if ids[i] in proteins_set:
            proteins_by_taxid[taxids[i]].append(ids[i])
    min_proteins_in_taxid = int(round(100 / (val_perc*100)))
    print('Min proteins for taxid specific sampling:', min_proteins_in_taxid)
    
    print('Indexing proteins by go id')
    ann_by_go = {}
    for protid, prot_ann in tqdm(annotations.items()):
        for goid in prot_ann:
            if not goid in ann_by_go:
                ann_by_go[goid] = set([protid])
            else:
                ann_by_go[goid].add(protid)

    print('Trying different splits')
    splits = []
    min_so_far = 300
    for _ in tqdm(range(600)):
        traintest_set, val_set = split_by_taxid(proteins_by_taxid, val_perc, 
                                                            min_proteins_in_taxid)
        faulty_go_ids = set()

        for goid, prot_ann in ann_by_go.items():
            local_traintest = prot_ann.intersection(traintest_set)
            local_val = prot_ann.intersection(val_set)
            if len(local_traintest) <= 3:
                #print(goid, 'has zero samples in traintest set')
                faulty_go_ids.add(goid)
            elif len(local_val) <= 3:
                #print(goid, 'has zero samples in validation set')
                faulty_go_ids.add(goid)
        if len(faulty_go_ids) < min_so_far:
            min_so_far = len(faulty_go_ids)
            print('Split with', len(faulty_go_ids), 'faulty GO ids')
        splits.append({'faulty_go_ids': faulty_go_ids, 
                        'traintest_set': traintest_set, 'val_set': val_set})
    
    splits.sort(key=lambda s: len(s['faulty_go_ids']))
    best_split = splits[0]
    traintest_set = best_split['traintest_set']
    val_set = best_split['val_set']
    faulty_go_ids = best_split['faulty_go_ids']

    val_sorted = []
    traintest_sorted = []
    for protein_id in ids:
        if protein_id in val_set:
            val_sorted.append(protein_id)
        elif protein_id in traintest_set:
            traintest_sorted.append(protein_id)
    
    print('Split:', len(traintest_sorted), len(val_sorted))
    
    return traintest_sorted, val_sorted, faulty_go_ids
