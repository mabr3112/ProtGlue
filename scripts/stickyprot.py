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
from protflow.tools.residue_selectors import ChainSelector, TrueSelector
from protflow.tools.colabfold import Colabfold, calculate_poses_interaction_pae
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

def compile_msa_str(b_seq, t_seq, target_msa_sequences) -> str:
    return NotImplemented

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
    motif_bb_rmsd = MotifRMSD(ref_col="rfdiff_1_location", jobstarter=sbatch_cpu_jobstarter)
    rosetta=Rosetta(jobstarter=sbatch_cpu_jobstarter)

    # setup chain selectors
    chain_selector = ChainSelector()
    true_selector = TrueSelector()
    chain_adder = ChainAdder(jobstarter=sbatch_cpu_jobstarter)
    sequence_remover = SequenceRemover(jobstarter=sbatch_cpu_jobstarter)

    # Load poses from the input directory
    poses = Poses(poses=args.target, work_dir=args.output_dir)

    # setup results directory
    results_dir = f"{poses.work_dir}/results/"
    os.makedirs(results_dir, exist_ok=True)

    # define target contig.
    true_selector.select("target_residues", poses)
    target_residues = poses.df["target_residues"].values[0].to_rfdiffusion_contig()
    target_length = len(target_residues)
    target_contig = args.target_contig or target_residues

    diff_opts = f"diffuser.T=50 'contigmap.contigs=[{target_contig}/0 {args.binder_length}-{args.binder_length}]' 'ppi.hotspot_res=[{args.hotspot_residues}]' potentials.guiding_potentials=[\\'type:custom_binder_potential,binderlen:80,contacts_weight:{args.contacts_weight},rog_weight:{args.rog_weight}\\']"

    # Diffuse binders to target
    logging.info(f"Starting diffusion of {args.num_diff} binders on {rfdiffusion_runner.jobstarter.max_cores} cores.")
    multiplex_poses = args.multiplex_rfdiffusion or None
    poses = rfdiffusion_runner.run(
        poses=poses,
        prefix="rfdiff_1",
        num_diffusions=args.num_diff,
        options=diff_opts,
        multiplex_poses=multiplex_poses
    )

    # calculate rfdiffusion binder stats
    poses.df["rfdiffusion_rog"] = [calc_rog_of_pdb(pose, chain="A") for pose in poses.poses_list()]
    poses.df["rfdiffusion_binder_contacts"] = [calc_interchain_contacts_pdb(pose, chains=["A", "B"]) for pose in poses.poses_list()]

    # plot rfdiffusion results
    plots.violinplot_multiple_cols(
        dataframe=poses,
        cols = ["rfdiff_1_plddt", "rfdiffusion_rog", "rfdiffusion_binder_contacts"],
        y_labels = ["pLDDT", "ROG [\u00C5]", "# contacting Ca"],
        titles = ["Rfdiffusion pLDDT", "ROG", "Contacts"],
        out_path = f"{results_dir}/rfdiffusion_stats.png"
    )

    # remove unusable diffusion outputs
    poses.filter_poses_by_value("rfdiff_1_plddt", args.rfdiff_plddt_cutoff, ">", prefix="rfdiff_plddt_filter", plot=True)
    poses.filter_poses_by_value("rfdiffusion_rog", args.rog_cutoff, "<", prefix="rfdiffusion_rog_filter", plot=True)
    poses.filter_poses_by_value("rfdiffusion_binder_contacts", args.contacts_cutoff, ">", prefix="rfdiffusion_binder_contacts_filter", plot=True)

    # combine binder stats into composite score
    poses.calculate_composite_score(f"rfdiffusion_custom_binder_score", scoreterms=["rfdiffusion_rog", "rfdiffusion_binder_contacts"], weights=[1, -1])
    poses.filter_poses_by_rank(0.8, score_col="rfdiffusion_custom_binder_score", prefix='filter_poses_by_rank', plot=True)

    # break if diffuse_only
    if args.diffuse_only:
        sys.exit(0)

    partial_diffusion_target_contig = f"B{int(args.binder_length)+1}-{int(args.binder_length) + target_length}"
    diff_opts = f"diffuser.partial_T=20 'contigmap.contigs=[{partial_diffusion_target_contig}/0 {args.binder_length}-{args.binder_length}]' 'ppi.hotspot_res=[{args.hotspot_residues.replace('A', 'B')}]'"

    logging.info("running partial diffusion")
    poses = rfdiffusion_runner.run(
        poses=poses,
        prefix="rfdiff_p",
        num_diffusions=args.num_partial_diffusions,
        options=diff_opts,
    )

    num_cycles = args.num_refinement_cycles  # For example, 3 cycles
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
        poses.filter_poses_by_rank(n=9, score_col='esm_1_plddt', remove_layers=1, ascending=False, prefix='top9_per_input', plot=True)
        motif_bb_rmsd.run(poses, prefix=f"rmsd_rfdiffusion_esm_bb_{cycle}", target_motif="esm_chain_A", ref_motif="rfdiffusion_binder_res", atoms=["C", "CA", "N", "O"])

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

        logging.info(f"converting all pdbs to fasta files for af prediction")
        poses.convert_pdb_to_fasta(prefix=f"cycle_{cycle}_fasta", update_poses=True)
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

## This is only executed when the script is started as a script
if __name__ == "__main__":
    # Setup argument parser
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

   # Required options
    argparser.add_argument('--target', type=str, required=True, help='Path to the pdb-file that should be used as target.')
    argparser.add_argument("--output_dir", type=str, required=True, help='Path to the folder containing the output')
    argparser.add_argument("--sequences", type=str, required=True, help='Number of sequences generated by LigandMPNN')

    # RFdiffusion Options
    argparser.add_argument("--binder_length", type=str, default="80", help='Specify the length of your binder')
    argparser.add_argument("--hotspot_residues", type=str, required=True, help='Specify the hotspot residues on your target that you want to target with your binder.')
    argparser.add_argument("--num_diff", type=int, default=100, help='Number of diffusion outputs')
    argparser.add_argument("--rog_cutoff", type=float, default=18, help="ROG cutoff to filter RFdiffusion outputs by. For binder lengths of 80 residues, we recommend to set this value to 14!")
    argparser.add_argument("--contacts_cutoff", type=float, default=300, help="Cutoff for interchain contact count after Rfdiffusion. For 80 residue binders, we typically use 300. The larger your binder, the larger this value can be. Check your output_dir/results/ folder for ROG and contacts distribution of your RFdiffusion output.")
    argparser.add_argument("--rfdiff_plddt_cutoff", type=float, default=0.92, help="pLDDT cutoff to discard RFdiffusion outputs.")
    argparser.add_argument("--multiplex_rfdiffusion", type=int, help="Number of parallel copies of RFdiffusion that should run (this is an efficiency option). Be aware that the total number of diffusion outptus will be 'num_diff' * 'multiplex_rfdiffusion'.")
    argparser.add_argument("--rog_weight", type=float, default=1, help="A higher value of this weight will result in more compact structures. Too high potential weights lead to undesignable backbones. Specify the weight on the ROG portion of the custom RFdiffusion binder potential.")
    argparser.add_argument("--contacts_weight", type=float, default=1, help="A higher value of this weight will result in more contacts to the binder. Too high potential weights lead to undesignable backbones. Specify the weight on the contacts portion of the custom RFdiffusion binder potential.")
    argparser.add_argument("--num_partial_diffusion", type=int, default=3, help="Number of partial diffusion runs to run.")
    argparser.add_argument("--diffuse_only", action="store_true", help="Specify whether to exit the script after diffusion.")

    # other
    argparser.add_argument("--num_refinement_cycles", default=3, type=int, help="Specify the number of refinement cycles of the binder design refinement you want to run after RFdiffusion. We recommend 3, usually refinement converges after latest 5 refinement cycles.")
    argparser.add_argument("--fastrelax_protocol", help='Specify default fastrelax_protocol to use in the Rosetta Relax steps. Defaults to ProtFlow\'s fastrelax protocol')
    argparser.add_argument("--fake_msa_path", type=str, help="path to fake target_msa to be used for AlphaFold2 predictions.")
    argparser.add_argument("--full_target_seq", type=str, help="Sequence of full target to use for prediction. DO NOT CHANGE THIS!")

    # Parse arguments
    arguments = argparser.parse_args()

    # Call the main function with parsed arguments
    main(arguments)
