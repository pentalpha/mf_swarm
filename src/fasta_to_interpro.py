import sys
from mf_swarm_lib.data.interpro_api.consult_interpro import consult_uniprot_ids
from os import path, remove

if __name__ == '__main__':
    #input fasta or txt file with uniprot ids
    input_file = sys.argv[1]
    #tsv file to save annotations
    output_file = sys.argv[2]
    results_cache = output_file + '.cache'

    if path.exists(results_cache+'.lock'):
        remove(results_cache+'.lock')

    ids = []
    with open(input_file, 'r') as infile:
        for line in infile:
            if 'fasta' in input_file:
                if line.startswith('>'):
                    newid = line.lstrip('>').split()[0].split('|')[0]
                    ids.append(newid)
            else:
                ids.append(line.strip())

    consult_uniprot_ids(ids, results_cache)

    #Copy cache to final output
    with open(results_cache, 'r') as cache_file, open(output_file, 'w') as out_file:
        for line in cache_file:
            out_file.write(line)