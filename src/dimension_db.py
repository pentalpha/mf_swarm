from os import path, stat
import json
from tqdm import tqdm

from util_base import run_command

protein_dimension_db_url = 'https://ucrania.imd.ufrn.br/~pitagoras/protein_dimension_db'

class DimensionDB:
    def __init__(self, dimension_db_releases_dir, dimension_db_release_n) -> None:
        print('Creating file names')
        self.release_dir        = path.join(dimension_db_releases_dir, 'release_' + dimension_db_release_n)
        self.release_url        = protein_dimension_db_url + '/release_'+ dimension_db_release_n
        self.plm_names          = ['ankh_base', 'ankh_large', 'esm2_t6', 'esm2_t12', 'esm2_t30', 'esm2_t33', 'esm2_t36', 'prottrans']
        self.emb_extension      = '.npy.gz'
        self.taxa_profile_names = ['taxa_profile_128', 'taxa_profile_256']
        self.taxa_onehot_names  = ['taxa_256', 'taxa_128']
        self.top_taxa_names     = ['top_taxa_128', 'top_taxa_256']

        self.feature_paths = {}
        for n in self.taxa_onehot_names:
            self.feature_paths[n] = path.join(self.release_dir, 'onehot.' + n + self.emb_extension)
        for n in self.taxa_profile_names:
            self.feature_paths[n] = path.join(self.release_dir, 'emb.' + n + self.emb_extension)
        for n in self.plm_names:
            self.feature_paths[n] = path.join(self.release_dir, 'emb.' + n + self.emb_extension)
        for n in self.top_taxa_names:
            self.feature_paths[n] = path.join(self.release_dir, n + '.txt')

        fasta_path = path.join(self.release_dir, 'uniprot_sorted.not_large.fasta')
        taxids_path = path.join(self.release_dir, 'taxid.tsv')
        ids_uniprot_path = path.join(self.release_dir, 'ids.txt')
        mf_gos_path = path.join(self.release_dir, 'go.experimental.mf.tsv.gz')
        self.other_files = [fasta_path, taxids_path, ids_uniprot_path, mf_gos_path]

        print('Checking for downloads')
        for p in tqdm(list(self.feature_paths.values()) + self.other_files):
            basename = path.basename(p)
            download_url = self.release_url + '/' + basename
            download_it = not path.exists(p)
            if not download_it:
                gzip_finished = run_command(['gzip -t', p], no_output=True) == 0 if p.endswith('.gz') else True
                if not gzip_finished or stat(p).st_size == 0:
                    download_it = True

            if download_it:
                print("Downloading", p)
                run_command(['cd', self.release_dir, '&&', 'wget', '-O', basename, download_url, '2> /dev/null'])
                if stat(p).st_size == 0:
                    run_command(['rm', p])
            else:
                print('Not downloading', p)

        print("Checking for missing files")
        for name in self.feature_paths.keys():
            if not path.exists(self.feature_paths[name]):
                print(self.feature_paths[name], 'does not exist')
                self.feature_paths[name] = None
        
        print('Available features:')
        print(json.dumps(self.feature_paths, indent=4))

        for i in range(len(self.other_files)):
            p = self.other_files[i]
            print(p, path.exists(p))
            if not path.exists(p):
                self.other_files[i] = None

        print("Loading ids")
        self.taxids = open(taxids_path, 'r').read().split('\n')
        self.taxids = [l.split("\t")[1] for l in self.taxids]
        self.ids = open(ids_uniprot_path, 'r').read().split('\n')