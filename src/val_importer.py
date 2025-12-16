import argparse
import sys
from mf_swarm_lib.importers import annopro, deepgose, interpro, mf_swarm, tale

def main():
    parser = argparse.ArgumentParser(description="Unified importer for validation results.")
    subparsers = parser.add_subparsers(dest='tool', required=True, help='Tool to import results from.')

    # Annopro
    parser_annopro = subparsers.add_parser('annopro')
    parser_annopro.add_argument('prot_dimension_db_release_path')
    parser_annopro.add_argument('min_protein_annots')
    parser_annopro.add_argument('val_part')
    parser_annopro.add_argument('annopro_res_dir')
    parser_annopro.add_argument('output_dir')

    # DeepGoSE
    parser_deepgose = subparsers.add_parser('deepgose')
    parser_deepgose.add_argument('prot_dimension_db_release_path')
    parser_deepgose.add_argument('min_protein_annots')
    parser_deepgose.add_argument('val_part')
    parser_deepgose.add_argument('deepgose_dfs_dir')
    parser_deepgose.add_argument('output_dir')

    # Interpro
    parser_interpro = subparsers.add_parser('interpro')
    parser_interpro.add_argument('prot_dimension_db_release_path')
    parser_interpro.add_argument('min_protein_annots')
    parser_interpro.add_argument('val_part')
    parser_interpro.add_argument('interpro_res')
    parser_interpro.add_argument('output_dir')

    # MF Swarm
    parser_mf_swarm = subparsers.add_parser('mf_swarm')
    parser_mf_swarm.add_argument('mf_swarm_trained_dir')
    parser_mf_swarm.add_argument('output_dir')

    # Tale
    parser_tale = subparsers.add_parser('tale')
    parser_tale.add_argument('prot_dimension_db_release_path')
    parser_tale.add_argument('min_protein_annots')
    parser_tale.add_argument('val_part')
    parser_tale.add_argument('tale_res_dir')
    parser_tale.add_argument('output_dir')

    args = parser.parse_args()

    if args.tool == 'annopro':
        annopro.run(args.prot_dimension_db_release_path, args.min_protein_annots, args.val_part, args.annopro_res_dir, args.output_dir)
    elif args.tool == 'deepgose':
        deepgose.run(args.prot_dimension_db_release_path, args.min_protein_annots, args.val_part, args.deepgose_dfs_dir, args.output_dir)
    elif args.tool == 'interpro':
        interpro.run(args.prot_dimension_db_release_path, args.min_protein_annots, args.val_part, args.interpro_res, args.output_dir)
    elif args.tool == 'mf_swarm':
        mf_swarm.run(args.mf_swarm_trained_dir, args.output_dir)
    elif args.tool == 'tale':
        tale.run(args.prot_dimension_db_release_path, args.min_protein_annots, args.val_part, args.tale_res_dir, args.output_dir)

if __name__ == "__main__":
    main()
