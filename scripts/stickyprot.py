#!/home/s1022006/miniconda3/envs/protflow/bin/python
## system
import sys
import random
import argparse
import os
import logging

# dependencies
from protflow.poses import Poses
from protflow.jobstarters import SbatchArrayJobstarter
from protflow.tools.rfdiffusion import RFdiffusion
from protflow.tools.ligandmpnn import LigandMPNN
from protflow.tools.esmfold import ESMFold
from protflow.metrics.rmsd import MotifRMSD
from protflow.utils.metrics import calc_rog_of_pdb
import protflow.utils.plotting as plots
from protflow.tools.residue_selectors import ChainSelector
from protflow.tools.colabfold import Colabfold
from protflow.tools.colabfold import calculate_poses_interaction_pae
from protflow.tools.protein_edits import SequenceRemover, ChainAdder
from protflow.utils.metrics import calc_interchain_contacts_pdb
from protflow.config import AUXILIARY_RUNNER_SCRIPTS_DIR as scripts_dir
from protflow.tools.rosetta import Rosetta
from protflow.utils.utils import parse_fasta_to_dict

# here we define miscellaneous functions
def random_numbers() -> str:
    '''Guess what it does.'''
    return "".join([str(x) for x in random.sample(range(1,10), 9)])

def get_target_msa(msa_path: str) -> list:
    '''Collects MSA info from a3m file. Has to be formatted customly as created in this notebook/script.'''    
    with open(msa_path, 'r', encoding="UTF-8") as f:
        raw_target_msa = f.read()

    # only extract sequences and remove colons
    target_msa = [x.strip().replace(":", "") for x in raw_target_msa.split("\n")[1:] if x]
    print(f"length of target msa: ", len(target_msa))
    return target_msa

def compile_dimer_msa_str(b_seq, t_seq, target_msa_sequences) -> str:
    '''Creates fake msa_str for alphafold prediction. very custom, not to be used by anyone.'''
    # compile gap strings for target and binder
    b_len = len(b_seq)
    t_len = len(t_seq)
    b_gap = "-"*b_len
    t_gap = '-'*t_len

    # write msa-str for target
    msa_target = "\n".join([f">target_{str(i).zfill(4)}\t{random_numbers()}\t{random_numbers()}\n{b_gap}{seq}" for i, seq in enumerate(target_msa_sequences, start=1)])

    # write str for colabfold_batch input fasta.
    msa_str = f"""#{b_len},{t_len},{t_len}\t1,1,1
>101\t102\t103
{b_seq}{t_seq}{t_seq}
>101
{b_seq}{t_gap}{t_gap}
>102
{b_gap}{t_seq}{t_gap}
{msa_target}
>103
{b_gap}{t_gap}{t_seq}
{msa_target}
"""
    return msa_str

## This defines what is done when the script is started as a script
def main(args):
    '''Main function that processes poses using LigandMPNN.'''
    # set logging
    import logging
    logging.basicConfig(
        filename=f'{args.output_dir}/rfdiffusion_protflow_log.txt',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # setup jobstarters
    sbatch_gpu_jobstarter = SbatchArrayJobstarter(max_cores=10, gpus=1)
    sbatch_cpu_jobstarter = SbatchArrayJobstarter(max_cores=470)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Create instances of necessary classes
    ligandmpnn_runner = LigandMPNN(jobstarter=sbatch_cpu_jobstarter)
    rfdiffusion_runner = RFdiffusion(jobstarter=sbatch_cpu_jobstarter)
    esmfold_runner = ESMFold(jobstarter=sbatch_gpu_jobstarter)
    colabfold_runner = Colabfold(jobstarter=sbatch_cpu_jobstarter)
    #template_bb_rmsd = BackboneRMSD(ref_col="rfdiff_1_location", chains=["D", "B"], jobstarter=sbatch_cpu_jobstarter)
    motif_bb_rmsd = MotifRMSD(ref_col="rfdiff_1_location", jobstarter=sbatch_cpu_jobstarter)
    rosetta=Rosetta(jobstarter=sbatch_cpu_jobstarter)

    # Load poses from the input directory
    poses = Poses(poses=args.input_dir, glob_suffix='*pdb', work_dir=args.output_dir)  # Corrected to use the actual argument
    #poses.set_logger()

    # setup chain selectors
    chain_selector = ChainSelector()
    chain_adder = ChainAdder(jobstarter=sbatch_cpu_jobstarter)
    sequence_remover = SequenceRemover(jobstarter=sbatch_cpu_jobstarter)

    #diff_opts = f"diffuser.T=50 'contigmap.contigs=[{args.contigmap} {args.binder_length}-{args.binder_length}]' 'ppi.hotspot_res=[{args.hotspot_residues}]'"
    diff_opts = f"diffuser.T=50 'contigmap.contigs=[A1-106/0 {args.binder_length}-{args.binder_length}]' 'ppi.hotspot_res=[{args.hotspot_residues}]' potentials.guiding_potentials=[\\'type:custom_binder_potential,binderlen:80,contacts_weight:1,rog_weight:1\\']"
    #potentials.guiding_potentials=[\\'type:binder_ROG,binderlen:80,weight:50,min_dist:0\\']
    # Show poses dataframe
    print(poses.df)

    # Run the LigandMPNN process with the provided arguments
    logging.info(f"Pre-diffusion")
    poses = rfdiffusion_runner.run(
        poses=poses,
        prefix="rfdiff_1",
        num_diffusions=int(args.num_diff),
        options=diff_opts,
        multiplex_poses=64
    )

    # calculate rfdiffusion binder stats
    poses.df["rfdiffusion_rog"] = [calc_rog_of_pdb(pose, chain="A") for pose in poses.poses_list()]
    poses.df["rfdiffusion_binder_contacts"] = [calc_interchain_contacts_pdb(pose, chains=["A", "B"]) for pose in poses.poses_list()]

    # combine binder stats into composite score
    poses.calculate_composite_score(f"rfdiffusion_custom_binder_score", scoreterms=["rfdiffusion_rog", "rfdiffusion_binder_contacts"], weights=[1, -1])
    poses.filter_poses_by_rank(0.8, score_col="rfdiffusion_custom_binder_score", prefix='filter_poses_by_rank', plot=True)

    results_dir = f"{poses.work_dir}/results/"
    os.makedirs(results_dir, exist_ok=True)

    print('original dataframe:')
    print(poses.df[['poses_description', 'rfdiff_1_plddt', 'rfdiffusion_rog', 'rfdiffusion_binder_contacts']])
    poses.filter_poses_by_value("rfdiff_1_plddt", 0.92, ">", prefix="rfdiff_plddt_filter", plot=True)
    poses.filter_poses_by_value("rfdiffusion_rog", 14, "<", prefix="rfdiffusion_rog_filter", plot=True)
    poses.filter_poses_by_value("rfdiffusion_binder_contacts", 300, ">", prefix="rfdiffusion_binder_contacts_filter", plot=True)
    print('filtered DataFrame:')
    print(poses.df[['poses_description', 'rfdiff_1_plddt', 'rfdiffusion_rog', 'rfdiffusion_binder_contacts']])

    results_dir = f"{poses.work_dir}/results/"
    os.makedirs(results_dir, exist_ok=True)

    diff_opts = f"diffuser.partial_T=20 'contigmap.contigs=[B81-186/0 {args.binder_length}-{args.binder_length}]' 'ppi.hotspot_res=[{args.hotspot_residues.replace('A', 'B')}]'"
    print(poses.df)

    logging.info("running partial diffusion")
    poses = rfdiffusion_runner.run(
        poses=poses,
        prefix="rfdiff_p",
        num_diffusions=3,
        options=diff_opts,
    )

    results_dir = f"{poses.work_dir}/results/"
    os.makedirs(results_dir, exist_ok=True)

    num_cycles = 3  # For example, 3 cycles

    print(poses.df["poses"].to_list())
    for cycle in range(1, num_cycles+1):
        print(f"Processing cycle {cycle}")

        # select RMSD ref_motif
        chain_selector.select("rfdiffusion_binder_res", poses=poses, chain="A")

        logging.info(f"running ligandmpnn")
         # Run the LigandMPNN process with the provided arguments
        poses = ligandmpnn_runner.run(
            poses=poses,
            prefix=f"ligandmpnn_{cycle}",
            nseq=int(args.sequences),  # Use the sequences argument
            model_type="ligand_mpnn",
            overwrite=False
        )

        logging.info(f"removing sequence of binder before esmfold preds")
        poses = sequence_remover.run(
            poses=poses,
            prefix=f"remove_seqs_{cycle}",
            chains=[1],
            overwrite=False
        )

        logging.info(f"Predicting {len(poses)} sequences with ESMFold")
        poses = esmfold_runner.run(
            poses=poses,
            prefix=f'esm_{cycle}'
        )

        chain_selector.select("esm_chain_A", poses=poses, chain="A")

        logging.info(f"printing")
        print('original dataframe:')
        print(poses.df[['poses_description', 'esm_1_plddt']])
        poses.filter_poses_by_rank(n=9, score_col='esm_1_plddt', remove_layers=1, ascending=False, prefix='top9_per_input', plot=True)
        print('filtered dataframe:')
        print(poses.df[['poses_description', 'esm_1_plddt']])

        motif_bb_rmsd.run(poses, prefix=f"rmsd_rfdiffusion_esm_bb_{cycle}", target_motif="esm_chain_A", ref_motif="rfdiffusion_binder_res", atoms=["C", "CA", "N", "O"])

        results_dir = f"{poses.work_dir}/results/"
        os.makedirs(results_dir, exist_ok=True)

        # 1. Define the columns, titles, and y-labels before the plotting function
        cols = ["esm_1_plddt", "rfdiff_1_plddt"]  # Replace with actual column names from poses.df
        titles = ["esm_1_plddt", "rfdiff_1_plddt"]  # Titles for the violin plots
        y_labels = ["esm_1_plddt", "rfdiff_1_plddt"]  # Y-axis l

        logging.info(f"plotting")
        plots.violinplot_multiple_cols(
            dataframe = poses.df,
            cols = cols, # specify the columns in poses.df that you would like to plot (as a list ["col_a", "col_b" ...]
            titles = titles, # specify the titles that the individual violins should get, as a list ["title_col_a", "title_col_b", ...] # type: ignore
            y_labels = y_labels, # y_labels for all the violins, same as above, list of labels ["...", ...]
            dims = None,
            out_path = f"{results_dir}/esmplddt_rfdiffplddt{cycle}.png", # create an output directory for your plots!
            show_fig = False # on the cluster, never show the figures! in Jupyter Notebooks you can feel free to show them.
        )

        ## Add a chain to the poses
        #added_chains = chain_adder.add_chain(
        #    poses=poses,
        #    prefix=f"add_binder_{cycle}",
        #    ref_col="rfdiff_1_location",
        #    copy_chain="B",
        #    overwrite=True
        #)

        logging.info(f"converting all pdbs to fasta files for af prediction")
        poses.convert_pdb_to_fasta(prefix=f"cycle_{cycle}_fasta", update_poses=True)

        # Run the ColabFold process with the provided arguments
        #if args.single_sequence_msa:
        #    opts = "--msa-mode single_sequence"
        #else:
        opts = "--num-models 1 --num-recycle 3"

        # filter poses down
        poses.filter_poses_by_rank(5, f"esm_{cycle}_plddt", remove_layers=2)
        logging.info(f"Filtered down poses to final selection of {len(poses)} sequences for AF2 dimer preds.")

        # prep input for fake-msa generation
        sequences = [list(parse_fasta_to_dict(pose).values())[0].strip() for pose in poses.poses_list()]
        descriptions = poses.df["poses_description"].to_list()

        logging.info(f"Prepping fake msa's")
        # read in target-msa for fake_msa generation
        target_msa = get_target_msa(args.fake_msa_path)
        full_target_seq = args.full_target_seq or "MRESKTLGAVQIMNGLFHIALGGLLMIPAGIYAPICVTVWYPLWGGIMYIISGSLLAATEKNSRKCLVKGKMIMNSLSLFAAISGMILSIMDILNIKISHFLKMESLNFIRAHTPYINIYNCEPANPSEKNSPSTQYCYSIQSLFLGILSVMLIFAFFQELVIAG"

        # create fake a3m files
        fake_a3m_file_list = []
        os.makedirs((fake_a3m_out_dir := f"{args.output_dir}/fake_a3m_files/"), exist_ok=True)
        for seq, description in zip(sequences, descriptions):
            msa_out_str = compile_dimer_msa_str(b_seq=seq, t_seq=full_target_seq, target_msa_sequences=target_msa)
            with open((outf := f"{fake_a3m_out_dir}/{description}.a3m"), 'w', encoding="UTF-8") as f:
                f.write(msa_out_str)
            fake_a3m_file_list.append(outf)


        # set fake a3m files as new poses
        poses.df["poses"] = fake_a3m_file_list

        # predict with colabfold
        logging.info(f"predicting interactions with Colabfod.")
        poses = colabfold_runner.run(
            poses=poses,
            prefix=f"af_{cycle}",
            options=opts
        )

        print('original dataframe:')
        calculate_poses_interaction_pae(f"af_{cycle}_binder", poses=poses, pae_list_col=f"af_{cycle}_pae_list", binder_start=1, binder_end=80, target_start=81, target_end=186)
        print(poses.df[['poses_description', f'af_{cycle}_binder_pae_interaction', f'af_{cycle}_iptm']])
        filtered_df = poses.df[(poses.df[f'af_{cycle}_binder_pae_interaction'] < 14) & (poses.df[f'af_{cycle}_iptm'] > 0.65)]
        print('filtered DataFrame:')
        print(filtered_df[['poses_description', f'af_{cycle}_binder_pae_interaction', f'af_{cycle}_iptm']])
        poses.df = filtered_df
        poses.filter_poses_by_rank(n=10, score_col=f'af_{cycle}_binder_pae_interaction', ascending=False, prefix=f'best_of_ipAE_{cycle}', plot=True)
        poses.filter_poses_by_rank(n=10 , score_col=f'af_{cycle}_iptm', ascending=False, prefix=f'best_of_ipTM_{cycle}', plot=True)
        print('filtered dataframe:')
        print(poses.df[['poses_description', f'af_{cycle}_binder_pae_interaction', f'af_{cycle}_iptm']])
        sys.exit(1)

        results_dir = f"{poses.work_dir}/results/"
        os.makedirs(results_dir, exist_ok=True)

        cols = ["f'af_{cycle}_binder_pae_interaction', f'af_{cycle}_iptm'"]  # Replace with actual column names from poses.df
        titles = ["f'af_{cycle}_binder_pae_interaction', f'af_{cycle}_iptm'"]  # Titles for the violin plots
        y_labels = ["f'af_{cycle}_binder_pae_interaction', f'af_{cycle}_iptm'"]  # Y-axis l

        plots.violinplot_multiple_cols(
            dataframe = poses.df,
            cols = cols, # specify the columns in poses.df that you would like to plot (as a list ["col_a", "col_b" ...]
            titles = titles, # specify the titles that the individual violins should get, as a list ["title_col_a", "title_col_b", ...] # type: ignore
            y_labels = y_labels, # y_labels for all the violins, same as above, list of labels ["...", ...]
            dims = None,
            out_path = f"{results_dir}/filtered_ipTM_ipAE{cycle}.png", # create an output directory for your plots!
            show_fig = False # on the cluster, never show the figures! in Jupyter Notebooks you can feel free to show them.
        )

        #relax with fastrelax protocol
        logging.info(f"starting_fastrelax")
        fastrelax_protocol = args.fastrelax_protocol or f"{scripts_dir}/fastrelax.sap.xml"
        rosetta_options = f"-parser:protocol {fastrelax_protocol} -beta"
        rosetta.run(poses=poses, rosetta_application="rosetta_scripts.default.linuxgccrelease", prefix="fastrelax", nstruct=15, options=rosetta_options)

        logging.info(f"Postrelax filter.")
        poses.filter_poses_by_rank(
            n=1,
            score_col="fastrelax_total_score",
            prefix="fastrelax_down"
        )

    # Access and process the results
    print(poses)

    # Print the custom text passed via arguments
    #print(arguments.print_text())

    # Optional: You can save results to the output directory, if necessary
    # For example:
    # output_file = os.path.join(arguments.output_dir, "results.txt")
    # with open(output_file, 'w') as f:
    #     f.write(str(results))

## This is only executed when the script is started as a script
if __name__ == "__main__":
    # Setup argument parser
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

   # Required options
    argparser.add_argument('--input_dir', type=str, required=True, help='Path to the folder containing input pdb files')
    argparser.add_argument("--output_dir", type=str, required=True, help='Path to the folder containing the output')
    argparser.add_argument("--sequences", type=str, required=True, help='Number of sequences generated')
    argparser.add_argument("--binder_length", type=str, default=80, help='length of binder')
    argparser.add_argument("--hotspot_residues", type=str, required=True, help='list of hotspot residues')
    argparser.add_argument("--num_diff", type=str, required=True, help='Number of diffusion outputs')
    argparser.add_argument("--contigmap", type=str, required=True, help='Name of chain and residues, that rfdiff uses (not the whole pdb)')
    argparser.add_argument("--single_sequence_msa", action="store_true", help='')
    argparser.add_argument("--fastrelax_protocol", help='')
    argparser.add_argument("--fake_msa_path", type=str, help="path to fake target_msa to be used for AlphaFold2 predictions.")
    argparser.add_argument("--full_target_seq", type=str, help="Sequence of full target to use for prediction. DO NOT CHANGE THIS!")

    # Parse arguments
    arguments = argparser.parse_args()

    # Call the main function with parsed arguments
    main(arguments)
